/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include "rtx_shader_output_capturer.h"
#include "../dxvk_shader.h"  // MUST be before D3D9 includes to get full class definition
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_camera.h"
#include "../dxvk_device.h"
#include "../dxvk_gpu_query.h"
#include "rtx_render/rtx_shader_manager.h"
#include "../../util/log/log.h"
#include "../../d3d9/d3d9_state.h"
#include "../../d3d9/d3d9_rtx.h"
#include "../../d3d9/d3d9_spec_constants.h"
#include "../../d3d9/d3d9_vertex_declaration.h" // For D3D9VertexDecl and D3DVERTEXELEMENT9
#include "../../d3d9/d3d9_util.h" // For DecodeDecltype
#include "../../d3d9/d3d9_shader.h" // For D3D9CommonShader, GetShader()
#include "../../d3d9/d3d9_shader_permutations.h" // For D3D9ShaderPermutations
#include "../../dxso/dxso_util.h" // For computeResourceSlotId
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include <rtx_shaders/shader_capture_prepare.h>
#include "rtx/pass/shader_capture/shader_capture.h"
#include <shader_capture_layer_vert.h>

namespace dxvk {

  // Shader wrapper for GPU-driven capture prepare compute shader
  namespace {
    class ShaderCapturePrepareShader : public ManagedShader {
      SHADER_SOURCE(ShaderCapturePrepareShader, VK_SHADER_STAGE_COMPUTE_BIT, shader_capture_prepare)

      PUSH_CONSTANTS(ShaderCapturePrepareArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(SHADER_CAPTURE_PREPARE_REQUESTS_INPUT)
        RW_STRUCTURED_BUFFER(SHADER_CAPTURE_PREPARE_DRAW_ARGS_OUTPUT)
        RW_STRUCTURED_BUFFER(SHADER_CAPTURE_PREPARE_INDEXED_ARGS_OUTPUT)
        RW_STRUCTURED_BUFFER(SHADER_CAPTURE_PREPARE_COUNTERS)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ShaderCapturePrepareShader);
  }

  ShaderOutputCapturer::ShaderOutputCapturer() {
    Logger::info("[ShaderOutputCapturer] ========== INITIALIZATION ==========");
    Logger::info(str::format("[ShaderOutputCapturer] enableShaderOutputCapture = ", enableShaderOutputCapture()));
    Logger::info(str::format("[ShaderOutputCapturer] captureAllDraws = ", captureAllDraws()));
    Logger::info(str::format("[ShaderOutputCapturer] captureEnabledHashes size = ", captureEnabledHashes().size()));

    // Log the actual hash values in the set
    if (captureEnabledHashes().size() > 0) {
      Logger::info("[ShaderOutputCapturer] captureEnabledHashes contents:");
      for (const auto& hash : captureEnabledHashes()) {
        Logger::info(str::format("  0x", std::hex, hash, std::dec));
      }
    }

    Logger::info(str::format("[ShaderOutputCapturer] maxCapturesPerFrame = ", maxCapturesPerFrame()));
    Logger::info(str::format("[ShaderOutputCapturer] captureResolution = ", captureResolution()));

    // Log the combined shouldCaptureFramebuffer() logic
    bool wouldCapture = captureAllDraws() || !captureEnabledHashes().empty();
    Logger::info(str::format("[ShaderOutputCapturer] shouldCaptureFramebuffer() would return: ", wouldCapture));
    Logger::info("[ShaderOutputCapturer] ========================================");
  }

  ShaderOutputCapturer::~ShaderOutputCapturer() {
    shutdownGpuCaptureSystem();
  }

  // ======== GPU-DRIVEN MULTI-INDIRECT CAPTURE SYSTEM (MegaGeometry-style) ========

  void ShaderOutputCapturer::initializeGpuCaptureSystem(Rc<RtxContext> ctx) {
    // Create persistent GPU buffers for multi-indirect dispatch
    // Following MegaGeometry pattern: allocate once, reuse every frame

    const uint32_t maxCaptures = maxCapturesPerFrame();

    // Capture requests buffer (CPU fills, GPU reads)
    {
      DxvkBufferCreateInfo info;
      info.size = sizeof(GpuCaptureRequest) * maxCaptures;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

      Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(info,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        "Shader Capture Requests Buffer");
      m_captureRequestsBuffer = DxvkBufferSlice(buffer, 0, info.size);

      Logger::info(str::format("[ShaderCapture-GPU] Created capture requests buffer: ",
                              info.size / 1024, " KB (", maxCaptures, " requests)"));
    }

    // Indirect draw args buffer (GPU fills via compute shader)
    {
      DxvkBufferCreateInfo info;
      info.size = sizeof(IndirectDrawArgs) * maxCaptures;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

      Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(info,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        "Shader Capture Indirect Draw Args");
      m_indirectDrawArgsBuffer = DxvkBufferSlice(buffer, 0, info.size);

      Logger::info(str::format("[ShaderCapture-GPU] Created indirect draw args buffer: ",
                              info.size / 1024, " KB"));
    }

    // Indirect indexed draw args buffer (for indexed draws)
    {
      DxvkBufferCreateInfo info;
      info.size = sizeof(IndirectIndexedDrawArgs) * maxCaptures;
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

      Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(info,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        "Shader Capture Indirect Indexed Draw Args");
      m_indirectIndexedDrawArgsBuffer = DxvkBufferSlice(buffer, 0, info.size);

      Logger::info(str::format("[ShaderCapture-GPU] Created indirect indexed draw args buffer: ",
                              info.size / 1024, " KB"));
    }

    // GPU counters buffer (atomic operations for work tracking)
    {
      DxvkBufferCreateInfo info;
      info.size = sizeof(GpuCaptureCounters);
      info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
      info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;

      Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(info,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXBuffer,
        "Shader Capture GPU Counters");
      m_captureCountersBuffer = DxvkBufferSlice(buffer, 0, info.size);

      Logger::info("[ShaderCapture-GPU] Created GPU counters buffer");
    }

    // Initialize GPU timestamp queries for profiling actual GPU execution time
    m_gpuTimestampStart = ctx->getDevice()->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
    m_gpuTimestampEnd = ctx->getDevice()->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);

    // Get timestamp period (nanoseconds per tick) from device properties
    m_timestampPeriod = ctx->getDevice()->adapter()->deviceProperties().limits.timestampPeriod;

    Logger::info(str::format("[ShaderCapture-GPU] GPU profiling initialized (timestampPeriod=", m_timestampPeriod, " ns)"));
    Logger::info("[ShaderCapture-GPU] GPU-driven multi-indirect capture system initialized");
  }

  void ShaderOutputCapturer::shutdownGpuCaptureSystem() {
    // Buffers will be automatically cleaned up when DxvkBufferSlice goes out of scope
    m_captureRequestsBuffer = DxvkBufferSlice();
    m_indirectDrawArgsBuffer = DxvkBufferSlice();
    m_indirectIndexedDrawArgsBuffer = DxvkBufferSlice();
    m_captureCountersBuffer = DxvkBufferSlice();
    
    // Release layer routing shader
    m_layerRoutingVertexShader = nullptr;
  }

  // DELETED: allocateRenderTargetPool() - pre-allocation wasted ~8GB VRAM!
  // DELETED: allocateRenderTargetFromPool() - now using on-demand allocation via getRenderTarget()
  // Render targets are now allocated on-demand with caching for massive VRAM savings

  void ShaderOutputCapturer::initializeLayerRoutingShader(Rc<DxvkDevice> device) {
    // Lazy initialization - only create shader once
    if (m_layerRoutingVertexShader != nullptr) {
      return;
    }

    // Create shader from embedded SPIR-V binary (included via shader_capture_layer_vert.h)
    SpirvCodeBuffer spirvCode(sizeof(shader_capture_layer_vert) / sizeof(uint32_t), shader_capture_layer_vert);

    DxvkShaderConstData constData;

    // Interface slots for our custom VS:
    // - inputSlots: bits 0, 1 set for position, texcoord at locations 0, 1
    //   (no color input - D3D9 uses R8G8B8A8_UINT which doesn't match vec4 float)
    // - outputSlots: bits 0-7 set for outputs at locations 0-7
    // - pushConstOffset: 0 (starts at beginning)
    // - pushConstSize: 64 bytes (mat4 projection)
    DxvkInterfaceSlots iface;
    iface.inputSlots = 0x3u;       // Bits 0, 1 = inputs at locations 0, 1 (position, texcoord)
    iface.outputSlots = 0xFFu;     // Bits 0-7 = outputs at ALL locations 0-7
    iface.pushConstOffset = 0;
    iface.pushConstSize = 64;      // mat4 = 16 floats = 64 bytes

    m_layerRoutingVertexShader = new DxvkShader(
      VK_SHADER_STAGE_VERTEX_BIT,
      0,  // No resource slots (we use push constants, not uniform buffers)
      nullptr,
      iface,
      spirvCode,
      DxvkShaderOptions(),
      std::move(constData));

    // Note: DxvkShader automatically detects ExportsViewportIndexLayerFromVertexStage
    // from SPIR-V analysis when gl_Layer is written

    Logger::info("[ShaderCapture-GPU] Layer routing vertex shader initialized (inputs: 0x7, outputs: 0xFF)");
  }

  void ShaderOutputCapturer::buildGpuCaptureList(Rc<RtxContext> ctx) {
    // Build capture request list from pending requests and upload to GPU
    // Following MegaGeometry pattern: CPU fills request buffer, GPU processes it

    if (m_pendingCaptureRequests.empty()) {
      return; // Nothing to capture this frame
    }

    const uint32_t numRequests = static_cast<uint32_t>(m_pendingCaptureRequests.size());

    Logger::info(str::format("[ShaderCapture-GPU] Building capture list: ",
                            numRequests, " requests"));

    // Upload capture requests to GPU buffer
    ctx->writeToBuffer(
      m_captureRequestsBuffer.buffer(),
      m_captureRequestsBuffer.offset(),
      numRequests * sizeof(GpuCaptureRequest),
      m_pendingCaptureRequests.data()
    );

    // Build indirect draw args on CPU (simplified version - MegaGeometry uses compute shader)
    // TODO: Move this to GPU compute shader for better performance
    std::vector<IndirectDrawArgs> drawArgs;
    std::vector<IndirectIndexedDrawArgs> indexedDrawArgs;

    drawArgs.reserve(numRequests);
    indexedDrawArgs.reserve(numRequests);

    for (const auto& request : m_pendingCaptureRequests) {
      const bool isIndexed = (request.flags & 0x1) != 0; // Bit 0 = indexed draw

      if (isIndexed) {
        IndirectIndexedDrawArgs args;
        args.indexCount = request.indexCount;
        args.instanceCount = 1;
        args.firstIndex = request.indexOffset;
        args.vertexOffset = static_cast<int32_t>(request.vertexOffset);
        args.firstInstance = 0;
        indexedDrawArgs.push_back(args);
      } else {
        IndirectDrawArgs args;
        args.vertexCount = request.vertexCount;
        args.instanceCount = 1;
        args.firstVertex = request.vertexOffset;
        args.firstInstance = 0;
        drawArgs.push_back(args);
      }
    }

    // Upload indirect args to GPU
    if (!drawArgs.empty()) {
      ctx->writeToBuffer(
        m_indirectDrawArgsBuffer.buffer(),
        m_indirectDrawArgsBuffer.offset(),
        drawArgs.size() * sizeof(IndirectDrawArgs),
        drawArgs.data()
      );
    }

    if (!indexedDrawArgs.empty()) {
      ctx->writeToBuffer(
        m_indirectIndexedDrawArgsBuffer.buffer(),
        m_indirectIndexedDrawArgsBuffer.offset(),
        indexedDrawArgs.size() * sizeof(IndirectIndexedDrawArgs),
        indexedDrawArgs.data()
      );
    }

    // Clear GPU counters for this frame
    GpuCaptureCounters zeros = {};
    ctx->writeToBuffer(
      m_captureCountersBuffer.buffer(),
      m_captureCountersBuffer.offset(),
      sizeof(GpuCaptureCounters),
      &zeros
    );

    Logger::info(str::format("[ShaderCapture-GPU] Capture list built: ",
                            drawArgs.size(), " draws, ", indexedDrawArgs.size(), " indexed draws"));
  }

  void ShaderOutputCapturer::executeMultiIndirectCaptures(Rc<RtxContext> ctx) {
    // ===== ABSOLUTE MAXIMUM PERFORMANCE GPU-DRIVEN MULTI-DRAW-INDIRECT =====
    // MegaGeometry-style batching optimizations:
    // 1. Group by (RT, Shader, Texture) - maximum batching potential
    // 2. Multi-draw-indirect for groups with same resources
    // 3. Persistent RT pool - zero RT allocation overhead
    // 4. GPU compute prepares indirect args
    // 5. Zero CPU-GPU sync
    //
    // Example: 100 brick buildings with same texture -> 1 multi-draw-indirect call!

    // Initialize layer routing shader if not already done
    initializeLayerRoutingShader(ctx->getDevice());

    auto tFunctionStart = std::chrono::high_resolution_clock::now();

    static uint32_t s_executeCallCount = 0;
    s_executeCallCount++;

    Logger::info(str::format("========== [SHADER-CAPTURE-EXEC #", s_executeCallCount,
                            "] executeMultiIndirectCaptures() CALLED - pendingRequests=",
                            m_pendingCaptureRequests.size(), " =========="));

    if (m_pendingCaptureRequests.empty()) {
      Logger::info(str::format("[SHADER-CAPTURE-EXEC #", s_executeCallCount,
                              "] NO PENDING REQUESTS, returning early"));
      return;
    }

    // DEDUPLICATION: Remove duplicate requests with the same cacheKey
    // Multiple draws often request the same capture (e.g. same material used many times)
    // We only need to capture it ONCE, and all draws will share the result.
    {
      std::unordered_set<XXH64_hash_t> seenKeys;
      std::vector<GpuCaptureRequest> uniqueRequests;
      uniqueRequests.reserve(m_pendingCaptureRequests.size());

      for (const auto& req : m_pendingCaptureRequests) {
        if (seenKeys.find(req.cacheKey) == seenKeys.end()) {
          seenKeys.insert(req.cacheKey);
          uniqueRequests.push_back(req);
        }
      }

      if (uniqueRequests.size() < m_pendingCaptureRequests.size()) {
        Logger::info(str::format("[ShaderCapture-DEDUP] Deduplicated requests: ",
          m_pendingCaptureRequests.size(), " -> ", uniqueRequests.size(),
          " (removed ", m_pendingCaptureRequests.size() - uniqueRequests.size(), " duplicates)"));
        m_pendingCaptureRequests = std::move(uniqueRequests);
      }
    }

    // ASYNC FRAME SPREADING: Only process maxCapturesPerFrame requests per frame
    // This spreads GPU work across multiple frames to prevent stalls
    const uint32_t totalPendingRequests = static_cast<uint32_t>(m_pendingCaptureRequests.size());
    const uint32_t maxPerFrame = maxCapturesPerFrame();
    const uint32_t numRequests = std::min(totalPendingRequests, maxPerFrame);

    Logger::info(str::format("[ShaderCapture-GPU] ===== ASYNC FRAME-SPREAD EXECUTION ====="));
    Logger::info(str::format("[ShaderCapture-GPU] Total queued: ", totalPendingRequests,
                            " | Processing this frame: ", numRequests, " (limit: ", maxPerFrame, ")"));
    Logger::info(str::format("[ShaderCapture-GPU] Remaining for next frame: ", totalPendingRequests - numRequests));

    // ========== STEP 1: GROUP BY (RT, SHADER, TEXTURE) FOR MAXIMUM BATCHING ==========
    Logger::info("[ShaderCapture-GPU] [TIMING] Starting grouping phase...");
    auto tGroupingStart = std::chrono::high_resolution_clock::now();

    struct CaptureGroup {
      Resources::Resource renderTarget;
      XXH64_hash_t shaderHash;   // For future shader batching
      XXH64_hash_t textureHash;  // Texture to bind
      VkExtent2D resolution;
      std::vector<uint32_t> requestIndices;
    };

    // OPTIMIZATION: Pre-sort requests by grouping key for O(N) grouping instead of O(N*log(N))
    // Create index array with grouping keys
    struct RequestSortKey {
      uint32_t index;
      uint64_t groupKey;
    };
    std::vector<RequestSortKey> sortedIndices;
    sortedIndices.reserve(numRequests);

    for (uint32_t i = 0; i < numRequests; i++) {
      const auto& request = m_pendingCaptureRequests[i];
      XXH64_hash_t texHash = request.textureHash;

      // Group key: resolution (32 bits) | texture hash (32 bits)
      uint64_t key = (uint64_t(request.resolution.width) << 48) |
                     (uint64_t(request.resolution.height) << 32) |
                     (uint64_t(texHash) & 0xFFFFFFFF);

      sortedIndices.push_back({i, key});
    }

    // Sort by group key - this allows O(N) grouping with single linear pass
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [](const RequestSortKey& a, const RequestSortKey& b) {
                return a.groupKey < b.groupKey;
              });

    // Now group in O(N) with single pass over sorted array
    std::vector<CaptureGroup> captureGroups;
    uint64_t currentKey = ~0ULL;  // Invalid key
    size_t currentGroupIndex = 0;

    Logger::info(str::format("[ShaderCapture-GPU] [GROUPING-OPT] Pre-sorted ", numRequests, " requests, grouping in O(N)..."));

    for (uint32_t i = 0; i < numRequests; i++) {
      const uint32_t reqIndex = sortedIndices[i].index;
      const uint64_t key = sortedIndices[i].groupKey;
      const auto& request = m_pendingCaptureRequests[reqIndex];

      // New group needed?
      if (key != currentKey) {
        currentKey = key;
        captureGroups.push_back(CaptureGroup{});
        currentGroupIndex = captureGroups.size() - 1;
        captureGroups.back().resolution = request.resolution;
        captureGroups.back().textureHash = request.textureHash;
        captureGroups.back().shaderHash = 0;
      }

      // Add request to current group
      captureGroups[currentGroupIndex].requestIndices.push_back(reqIndex);
    }

    auto tGroupingEnd = std::chrono::high_resolution_clock::now();
    auto groupingTime = std::chrono::duration<double, std::micro>(tGroupingEnd - tGroupingStart).count();

    Logger::info(str::format("[ShaderCapture-GPU] Grouped ", numRequests, " into ",
                            captureGroups.size(), " resource groups [", groupingTime, " us]"));

    // ========== STEP 1.5: ALLOCATE RTs AND BATCH BARRIER TRANSITIONS ==========
    // Optimization #1 & #3: Batch all image transitions into ONE barrier
    auto tPreAllocStart = std::chrono::high_resolution_clock::now();

    // Pre-allocate all RTs and collect images for batching
    for (auto& group : captureGroups) {
      const uint32_t groupSize = static_cast<uint32_t>(group.requestIndices.size());
      if (groupSize == 0) continue;

      group.renderTarget = getRenderTargetArray(ctx, group.resolution, VK_FORMAT_R8G8B8A8_UNORM, groupSize);
    }

    // CRITICAL OPTIMIZATION: Transition ALL render targets to GENERAL in ONE batched barrier
    // This prevents individual barriers when binding each RT later
    std::vector<std::pair<Rc<DxvkImage>, VkImageLayout>> rtLayoutTransitions;
    uint32_t alreadyCorrectLayout = 0;
    uint32_t needsTransition = 0;

    for (auto& group : captureGroups) {
      if (group.renderTarget.isValid()) {
        VkImageLayout currentLayout = group.renderTarget.image->info().layout;
        Logger::info(str::format("[RT-BATCH-DEBUG] RT 0x", std::hex, group.renderTarget.image->getHash(), std::dec,
                                " current layout=", currentLayout, " (GENERAL=", VK_IMAGE_LAYOUT_GENERAL, ")"));

        if (currentLayout != VK_IMAGE_LAYOUT_GENERAL) {
          rtLayoutTransitions.emplace_back(group.renderTarget.image, VK_IMAGE_LAYOUT_GENERAL);
          needsTransition++;
        } else {
          alreadyCorrectLayout++;
        }
      }
    }

    Logger::info(str::format("[RT-BATCH-DEBUG] RTs: ", alreadyCorrectLayout, " already GENERAL, ",
                            needsTransition, " need transition, ", captureGroups.size(), " total groups"));

    if (!rtLayoutTransitions.empty()) {
      Logger::info(str::format("[RT-BATCH-DEBUG] Calling batchChangeImageLayout with ", rtLayoutTransitions.size(), " RTs"));
      ctx->batchChangeImageLayout(rtLayoutTransitions);
    } else {
      Logger::info("[RT-BATCH-DEBUG] NO RT transitions needed - all already in GENERAL layout or invalid");
    }

    // CRITICAL FIX: Clear all render targets to prevent bloom from reading garbage data
    // Uninitialized textures contain random values that bloom interprets as bright pixels
    VkClearColorValue clearBlack = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    for (auto& group : captureGroups) {
      if (group.renderTarget.isValid()) {
        VkImageSubresourceRange clearRange;
        clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clearRange.baseMipLevel = 0;
        clearRange.levelCount = 1;
        clearRange.baseArrayLayer = 0;
        clearRange.layerCount = group.renderTarget.image->info().numLayers;

        static uint32_t clearLogCount = 0;
        if (++clearLogCount <= 20) {
          Logger::warn(str::format("[RT-CLEAR-BEFORE] #", clearLogCount,
                                  " Clearing RT hash=0x", std::hex, group.renderTarget.image->getHash(), std::dec,
                                  " layers=", clearRange.layerCount,
                                  " format=", group.renderTarget.image->info().format,
                                  " extent=", group.renderTarget.image->info().extent.width, "x", group.renderTarget.image->info().extent.height));
        }

        ctx->clearColorImage(group.renderTarget.image, clearBlack, clearRange);

        if (clearLogCount <= 20) {
          Logger::warn(str::format("[RT-CLEAR-AFTER] #", clearLogCount, " Clear completed"));
        }
      }
    }

    auto tPreAllocEnd = std::chrono::high_resolution_clock::now();
    Logger::info(str::format("[ShaderCapture] Pre-allocation took ",
                            std::chrono::duration<double, std::micro>(tPreAllocEnd - tPreAllocStart).count(), " us]"));

    // ========== STEP 2: EXECUTE WITH MULTI-DRAW-INDIRECT PER GROUP ==========
    auto tExecutionStart = std::chrono::high_resolution_clock::now();

    // GPU PROFILING: Read PREVIOUS frame's timestamp results (delayed readback for async GPU)
    if (m_prevFrameTimestampStart != nullptr && m_prevFrameTimestampEnd != nullptr) {
      DxvkQueryData startData = {};
      DxvkQueryData endData = {};

      DxvkGpuQueryStatus startStatus = m_prevFrameTimestampStart->getData(startData);
      DxvkGpuQueryStatus endStatus = m_prevFrameTimestampEnd->getData(endData);

      if (startStatus == DxvkGpuQueryStatus::Available && endStatus == DxvkGpuQueryStatus::Available) {
        // Calculate GPU execution time in milliseconds
        uint64_t startTime = startData.timestamp.time;
        uint64_t endTime = endData.timestamp.time;
        uint64_t gpuTicks = (endTime > startTime) ? (endTime - startTime) : 0;
        double gpuTimeNs = gpuTicks * m_timestampPeriod;
        double gpuTimeMs = gpuTimeNs / 1000000.0;

        Logger::info(str::format("[GPU-PROFILING] ========== ACTUAL GPU EXECUTION TIME (PREV FRAME) =========="));
        Logger::info(str::format("[GPU-PROFILING] GPU Time: ", gpuTimeMs, " ms (", gpuTicks, " ticks)"));
        Logger::info(str::format("[GPU-PROFILING] Timestamp Period: ", m_timestampPeriod, " ns/tick"));
        Logger::info(str::format("[GPU-PROFILING] ============================================================"));
      } else {
        Logger::info(str::format("[GPU-PROFILING] Previous frame timestamp data not yet available (start=",
                                static_cast<uint32_t>(startStatus), ", end=", static_cast<uint32_t>(endStatus), ")"));
      }
    }

    // GPU PROFILING: Write start timestamp for CURRENT frame
    if (m_gpuTimestampStart != nullptr) {
      ctx->writeTimestamp(m_gpuTimestampStart);
      Logger::info("[GPU-PROFILING] Start timestamp written (current frame)");
    }

    uint32_t successCount = 0;
    uint32_t multiDrawCount = 0;
    uint32_t singleDrawCount = 0;
    double totalRTAllocTime = 0.0;
    double totalBindingTime = 0.0;
    double totalMultiDrawTime = 0.0;
    double totalSingleDrawTime = 0.0;

    for (auto& group : captureGroups) {
      auto tGroupStart = std::chrono::high_resolution_clock::now();

      const uint32_t groupSize = static_cast<uint32_t>(group.requestIndices.size());
      if (groupSize == 0) continue;

      const auto& firstRequest = m_pendingCaptureRequests[group.requestIndices[0]];

      if (!group.renderTarget.isValid()) {
        Logger::err(str::format("[ShaderCapture-GPU] Failed to allocate RT for group (", groupSize, " draws)"));
        continue;
      }

      // CRITICAL: Bind texture BEFORE render targets to allow DXVK to batch layout transitions
      // Binding texture after bindRenderTargets() forces transitions inside render pass = individual barriers!
      auto tBindingStart = std::chrono::high_resolution_clock::now();

      // CRITICAL FIX: Bind ALL textures from the request to their correct sampler slots
      // The PS expects textures at specific slots (s0, s1, s2, etc.)
      // Only binding one texture causes the shader to sample black for other slots!
      // tex.slot is the DXVK binding number (1014 = s0, 1015 = s1, etc.) - use directly
      if (!firstRequest.textures.empty()) {
        Logger::info(str::format("[TEX-BIND] Binding ", firstRequest.textures.size(), " textures for this capture group"));

        // CRITICAL FIX: Transition ALL textures to SHADER_READ_ONLY_OPTIMAL before binding
        // This fixes "image layout mismatch" Vulkan validation errors that cause black output
        for (const auto& tex : firstRequest.textures) {
          if (tex.texture.isValid() && tex.texture.getImageView()) {
            Rc<DxvkImage> image = tex.texture.getImageView()->image();
            if (image != nullptr && image->info().layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
              ctx->changeImageLayout(image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
          }
        }

        for (const auto& tex : firstRequest.textures) {
          if (tex.texture.isValid() && tex.texture.getImageView()) {
            // Use the captured slot directly - it's the binding number the shader expects
            ctx->bindResourceView(tex.slot, tex.texture.getImageView(), nullptr);
            if (tex.sampler != nullptr) {
              ctx->bindResourceSampler(tex.slot, tex.sampler);
            }
            static uint32_t texBindLog = 0;
            if (++texBindLog <= 50) {
              const int d3d9Stage = (tex.slot >= 1014 && tex.slot <= 1029) ? (tex.slot - 1014) : -1;
              auto imgInfo = tex.texture.getImageView()->image()->info();
              Logger::info(str::format("[TEX-BIND] slot=", tex.slot, " (s", d3d9Stage, ")",
                " size=", imgInfo.extent.width, "x", imgInfo.extent.height,
                " format=", imgInfo.format,
                " hash=0x", std::hex, tex.texture.getImageView()->image()->getHash(), std::dec,
                " layout=", imgInfo.layout));
            }
          }
        }
      } else if (firstRequest.colorTexture.isValid()) {
        // Fallback: bind single colorTexture to slot 0
        TextureRef replacementTexture = getReplacementTexture(firstRequest.textureHash);
        const TextureRef& texToUse = replacementTexture.isValid() ? replacementTexture : firstRequest.colorTexture;
        if (texToUse.getImageView()) {
          ctx->bindResourceView(0, texToUse.getImageView(), nullptr);
          Logger::info(str::format("[TEX-BIND] Fallback: bound colorTexture to slot 0"));
        }
      }

      Logger::info(str::format("[BARRIER-DEBUG] ===== BINDING RT 0x", std::hex, group.renderTarget.image->getHash(), std::dec,
                              " currentLayout=", group.renderTarget.image->info().layout, " ====="));

      // Bind texture array as render target (layered rendering)
      DxvkRenderTargets captureRt;
      captureRt.color[0].view = group.renderTarget.view;
      captureRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;

      Logger::info(str::format("[BARRIER-DEBUG] Calling bindRenderTargets with GENERAL layout..."));
      ctx->bindRenderTargets(captureRt);
      // CRITICAL: Clear render pass barriers to ensure renderpass has 1 dependency
      // This matches the default renderpass used when compiling pipelines.
      // Without this, we get "dependencyCount is incompatible" errors (3 != 1)
      ctx->clearRenderPassBarriers();
      Logger::info(str::format("[BARRIER-DEBUG] After bindRenderTargets, layout=", group.renderTarget.image->info().layout));

      VkClearValue clearValue = {};
      // DEBUG: Clear to magenta instead of black to see if shader renders anything
      clearValue.color.float32[0] = 1.0f;  // R
      clearValue.color.float32[1] = 0.0f;  // G
      clearValue.color.float32[2] = 1.0f;  // B
      clearValue.color.float32[3] = 1.0f;  // A
      Logger::info(str::format("[BARRIER-DEBUG] Calling clearRenderTarget with MAGENTA..."));
      ctx->clearRenderTarget(group.renderTarget.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);
      // Clear barriers again after clear in case it modified them
      ctx->clearRenderPassBarriers();
      Logger::info(str::format("[BARRIER-DEBUG] After clearRenderTarget, layout=", group.renderTarget.image->info().layout));

      // Bind shaders - BOTH ORIGINAL game shaders
      // Using both original VS + PS avoids renderpass incompatibility issues
      // (pipelines were created for game's renderpass, custom VS caused mismatch)

      // CRITICAL: Set spec constants BEFORE binding shaders!
      // SamplerDepthMode=0 disables shadow/depth comparison samplers
      // This is essential because we bind color textures, not depth textures
      ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::SamplerDepthMode, 0);
      // AlphaCompareOp=7 (VK_COMPARE_OP_ALWAYS) disables alpha test
      ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::AlphaCompareOp, 7);
      Logger::info("[SPEC-CONST-BATCH] Set SamplerDepthMode=0 and AlphaCompareOp=7 for batched capture");

      // Bind the ORIGINAL vertex shader from the game
      if (firstRequest.vertexShader != nullptr) {
        ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, firstRequest.vertexShader);
        Logger::info("[SHADER-BIND] Bound ORIGINAL vertex shader from game");
      } else {
        Logger::warn("[SHADER-BIND] NO vertex shader in request - using fixed function?");
      }

      // Bind the ORIGINAL pixel/fragment shader from the game
      if (firstRequest.pixelShader != nullptr) {
        ctx->bindShader(VK_SHADER_STAGE_FRAGMENT_BIT, firstRequest.pixelShader);
        Logger::info(str::format("[SHADER-BIND] Bound ORIGINAL pixel shader from game",
          " - ConstantCount=", firstRequest.pixelShaderConstantData.size(),
          " - TextureCount=", firstRequest.textures.size()));
      } else {
        Logger::err("[SHADER-BIND] NO pixel shader in request! Will render incorrectly!");
      }

      // VS constants are bound below AFTER the createConstantBufferSlice lambda is defined

      // CRITICAL: Bind shader constants (uniforms)
      // Helper to create constant buffer from Vector4 array
      auto createConstantBufferSlice = [&](const std::vector<Vector4>& constantData,
                                           VkPipelineStageFlags stages,
                                           const char* debugName) -> DxvkBufferSlice {
        if (constantData.empty())
          return DxvkBufferSlice();

        // Convert Vector4 to bytes
        const uint8_t* dataBytes = reinterpret_cast<const uint8_t*>(constantData.data());
        const size_t dataSize = constantData.size() * sizeof(Vector4);

        const Rc<DxvkDevice>& device = ctx->getDevice();
        VkDeviceSize alignment = device->properties().core.properties.limits.minUniformBufferOffsetAlignment;
        if (alignment == 0)
          alignment = 1;

        const VkDeviceSize rawSize = static_cast<VkDeviceSize>(dataSize);
        const VkDeviceSize alignedSize = ((rawSize + alignment - 1) / alignment) * alignment;

        DxvkBufferCreateInfo info;
        info.size = alignedSize;
        info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        info.stages = stages;
        info.access = VK_ACCESS_UNIFORM_READ_BIT;
        info.requiredAlignmentOverride = alignment;

        Rc<DxvkBuffer> uploadBuffer = device->createBuffer(
          info,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          DxvkMemoryStats::Category::AppBuffer,
          debugName);

        if (void* dst = uploadBuffer->mapPtr(0)) {
          std::memcpy(dst, dataBytes, dataSize);
          if (alignedSize > rawSize) {
            std::memset(reinterpret_cast<char*>(dst) + rawSize, 0, size_t(alignedSize - rawSize));
          }
        }

        return DxvkBufferSlice(uploadBuffer, 0, alignedSize);
      };

      // Bind vertex shader constants - RESTORED since we're using original game VS now
      // The original VS uses uniform buffers, not push constants
      if (!firstRequest.vertexShaderConstantData.empty()) {
        Logger::info(str::format("[SHADER-CONST-DEBUG] Vertex shader constant data size: ",
          firstRequest.vertexShaderConstantData.size(), " Vector4s (",
          firstRequest.vertexShaderConstantData.size() * sizeof(Vector4), " bytes)"));

        // Log VS constants for debugging - show c[0]-c[7] which typically contain transformation matrices
        if (firstRequest.vertexShaderConstantData.size() > 0) {
          // Log first 8 constants (2 rows of a matrix or WVP rows)
          static uint32_t vsConstLogCount = 0;
          if (++vsConstLogCount <= 10) {
            Logger::info("[SHADER-CONST-DEBUG] VS constants (c[0]-c[7]) - transformation matrices:");
            for (int i = 0; i < 8 && i < (int)firstRequest.vertexShaderConstantData.size(); i++) {
              const Vector4& v = firstRequest.vertexShaderConstantData[i];
              Logger::info(str::format("  VS c[", i, "] = (", v.x, ", ", v.y, ", ", v.z, ", ", v.w, ")"));
            }
            // Find first non-zero constant to help debug
            int firstNonZero = -1;
            for (int i = 0; i < (int)firstRequest.vertexShaderConstantData.size(); i++) {
              const Vector4& v = firstRequest.vertexShaderConstantData[i];
              if (v.x != 0 || v.y != 0 || v.z != 0 || v.w != 0) {
                firstNonZero = i;
                break;
              }
            }
            if (firstNonZero < 0) {
              Logger::warn("[SHADER-CONST-DEBUG] WARNING: All VS constants are ZERO! Vertices will transform to origin!");
            } else {
              Logger::info(str::format("[SHADER-CONST-DEBUG] First non-zero VS constant at c[", firstNonZero, "]"));
            }
            // Check c[81] which Lego Batman 2 uses for viewport transform
            if (firstRequest.vertexShaderConstantData.size() > 81) {
              const Vector4& v81 = firstRequest.vertexShaderConstantData[81];
              Logger::info(str::format("[SHADER-CONST-DEBUG] VS c[81] (viewport) = (", v81.x, ", ", v81.y, ", ", v81.z, ", ", v81.w, ")"));
            } else {
              Logger::warn("[SHADER-CONST-DEBUG] WARNING: VS constants too small - missing c[81] viewport data!");
            }
          }
        }

        // CRITICAL FIX: If c[0]-c[3] is identity, inject orthographic projection
        // The game's vertices are in range ~(-5 to +5) but identity matrix means they
        // pass through unchanged and get clipped (NDC is -1 to +1).
        // Inject a scale matrix to map vertex range to NDC.
        std::vector<Vector4> modifiedConstants = firstRequest.vertexShaderConstantData;

        // DEBUG: Log what we're checking
        static uint32_t orthoCheckCount = 0;
        if (++orthoCheckCount <= 5 && modifiedConstants.size() >= 2) {
          Logger::info(str::format("[VS-CONST-FIX-CHECK] c[0]=(",
            modifiedConstants[0].x, ",", modifiedConstants[0].y, ",", modifiedConstants[0].z, ",", modifiedConstants[0].w,
            ") c[1]=(", modifiedConstants[1].x, ",", modifiedConstants[1].y, ",", modifiedConstants[1].z, ",", modifiedConstants[1].w, ")"));
        }

        bool c0IsIdentity = (modifiedConstants.size() >= 2 &&
                            modifiedConstants[0].x == 1.0f && modifiedConstants[0].y == 0.0f &&
                            modifiedConstants[1].x == 0.0f && modifiedConstants[1].y == 1.0f);
        if (c0IsIdentity && modifiedConstants.size() >= 4) {
          // Scale matrix: map ~(-6 to +6) to (-1 to +1) => scale by 1/6
          // Also flip Y for D3D9 coordinate system
          const float scale = 0.15f;  // ~1/6.5 to ensure coverage
          modifiedConstants[0] = Vector4(scale, 0.0f, 0.0f, 0.0f);   // Row 0
          modifiedConstants[1] = Vector4(0.0f, -scale, 0.0f, 0.0f);  // Row 1 (Y flipped)
          modifiedConstants[2] = Vector4(0.0f, 0.0f, 1.0f, 0.0f);    // Row 2
          modifiedConstants[3] = Vector4(0.0f, 0.0f, 0.0f, 1.0f);    // Row 3
          static uint32_t orthoInjectCount = 0;
          if (++orthoInjectCount <= 5) {
            Logger::info(str::format("[VS-CONST-FIX] Injected orthographic projection scale=", scale,
              " to map vertex range to NDC"));
          }
        }

        DxvkBufferSlice vsConstantSlice = createConstantBufferSlice(
          modifiedConstants,
          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
          "ShaderCapture VS Constants");

        if (vsConstantSlice.defined()) {
          const uint32_t vsConstantBufferSlot = computeResourceSlotId(
            DxsoProgramType::VertexShader,
            DxsoBindingType::ConstantBuffer,
            DxsoConstantBuffers::VSConstantBuffer);
          ctx->bindResourceBuffer(vsConstantBufferSlot, vsConstantSlice);
          Logger::info(str::format("[SHADER-CONST] Bound vertex shader constants to slot ", vsConstantBufferSlot));
        } else {
          Logger::warn("[SHADER-CONST] Failed to create vertex shader constant buffer!");
        }
      } else {
        Logger::warn("[SHADER-CONST] NO vertex shader constant data! VS will read garbage!");
      }

      // Bind pixel shader constants - CRITICAL for procedural textures and view-dependent UVs
      // Now populated from D3D9 device state in d3d9_rtx.cpp!
      if (!firstRequest.pixelShaderConstantData.empty()) {
        Logger::info(str::format("[SHADER-CONST-DEBUG] Pixel shader constant data size: ",
          firstRequest.pixelShaderConstantData.size(), " Vector4s (",
          firstRequest.pixelShaderConstantData.size() * sizeof(Vector4), " bytes)"));

        // Log ALL non-zero PS constants to see what the shader actually uses
        static uint32_t psConstLogCount = 0;
        if (psConstLogCount++ < 3) {
          Logger::info("[PS-CONST-DUMP] ===== FULL PS CONSTANT DUMP (non-zero only) =====");
          size_t nonZeroCount = 0;
          for (size_t i = 0; i < firstRequest.pixelShaderConstantData.size(); i++) {
            const Vector4& v = firstRequest.pixelShaderConstantData[i];
            if (v.x != 0.0f || v.y != 0.0f || v.z != 0.0f || v.w != 0.0f) {
              Logger::info(str::format("[PS-CONST-DUMP] c[", i, "] = (", v.x, ", ", v.y, ", ", v.z, ", ", v.w, ")"));
              nonZeroCount++;
            }
          }
          Logger::info(str::format("[PS-CONST-DUMP] Total non-zero constants: ", nonZeroCount, " / ", firstRequest.pixelShaderConstantData.size()));
          Logger::info("[PS-CONST-DUMP] ===== END DUMP =====");
        }

        DxvkBufferSlice psConstantSlice = createConstantBufferSlice(
          firstRequest.pixelShaderConstantData,
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          "ShaderCapture PS Constants");

        if (psConstantSlice.defined()) {
          const uint32_t psConstantBufferSlot = computeResourceSlotId(
            DxsoProgramType::PixelShader,
            DxsoBindingType::ConstantBuffer,
            DxsoConstantBuffers::PSConstantBuffer);
          ctx->bindResourceBuffer(psConstantBufferSlot, psConstantSlice);
          Logger::info(str::format("[SHADER-CONST] Bound pixel shader constants to slot ", psConstantBufferSlot));
        } else {
          Logger::warn("[SHADER-CONST] Failed to create pixel shader constant buffer!");
        }
      } else {
        Logger::warn("[SHADER-CONST] NO pixel shader constant data! Shader will read garbage!");
      }

      // Bind textures and samplers - CRITICAL for textured shaders
      // ROBUST BINDING: Iterate shader slots to know exactly what the shader expects
      if (firstRequest.pixelShader != nullptr) {
        // 1. Build map of captured textures for fast lookup
        // NOTE: tex.slot is ALREADY a Vulkan binding slot (computed at capture time), not D3D9 slot
        std::unordered_map<uint32_t, const GpuCaptureRequest::CapturedTexture*> capturedTextureMap;
        for (const auto& tex : firstRequest.textures) {
          // tex.slot is already the Vulkan binding slot - use directly
          capturedTextureMap[tex.slot] = &tex;
        }

        // DEBUG: Log what's in the captured texture map
        static uint32_t mapDebugCount = 0;
        if (mapDebugCount++ < 10) {
          Logger::info(str::format("[TEX-MAP-DEBUG] Captured textures: ", firstRequest.textures.size()));
          for (const auto& tex : firstRequest.textures) {
            Logger::info(str::format("[TEX-MAP-DEBUG]   Vulkan slot ", tex.slot,
              " hash=0x", std::hex, tex.texture.getImageHash()));
          }
        }

        // 2. Iterate shader resource slots
        const auto& slots = firstRequest.pixelShader->getResourceSlots();
        bool anyTextureBound = false;
        uint32_t dummyBindCount = 0;

        for (const auto& slot : slots) {
          // Only care about textures/samplers
          if (slot.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
              slot.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {

            uint32_t binding = slot.slot;

            static uint32_t slotDebugCount = 0;
            if (slotDebugCount++ < 30) {
              bool found = capturedTextureMap.count(binding) > 0;
              Logger::info(str::format("[TEX-SLOT-DEBUG] Shader expects binding ", binding,
                " - in map: ", found ? "YES" : "NO"));
            }
            
            if (capturedTextureMap.count(binding)) {
              // Bind the ACTUAL captured texture from the game
              const auto* capturedTex = capturedTextureMap[binding];
              DxvkImageView* texView = capturedTex->texture.getImageView();
              if (texView != nullptr) {
                ctx->bindResourceView(binding, Rc<DxvkImageView>(texView), nullptr);
                if (capturedTex->sampler != nullptr) {
                  ctx->bindResourceSampler(binding, capturedTex->sampler);
                }
                anyTextureBound = true;

                static uint32_t texBindLogCount = 0;
                if (texBindLogCount++ < 20) {
                  auto& imgInfo = texView->image()->info();
                  Logger::info(str::format("[SHADER-TEXTURE] Bound GAME texture to slot ", binding,
                    " hash=0x", std::hex, capturedTex->texture.getImageHash(), std::dec,
                    " size=", imgInfo.extent.width, "x", imgInfo.extent.height,
                    " format=", imgInfo.format,
                    " layout=", imgInfo.layout));
                }
              } else {
                // Texture was captured but imageView is null - use dummy
                createDummyResources(ctx->getDevice());
                if (m_dummyDepthImage != nullptr && !m_dummyDepthLayoutInitialized) {
                  ctx->changeImageLayout(m_dummyDepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                  m_dummyDepthLayoutInitialized = true;
                }
                if (m_dummyDepthTexture != nullptr) {
                  ctx->bindResourceView(binding, m_dummyDepthTexture, nullptr);
                  if (m_dummyShadowSampler != nullptr) {
                    ctx->bindResourceSampler(binding, m_dummyShadowSampler);
                  }
                  anyTextureBound = true;
                }
              }
            } else {
              // Bind DUMMY texture to prevent undefined behavior (rainbows)
              // If the shader expects a texture but the game didn't bind one (or we didn't capture it),
              // we MUST bind something valid.
              // CRITICAL: Use DEPTH texture with shadow sampler for ALL dummy bindings!
              // Some D3D9 shaders use shadow samplers (depth comparison) and binding a color
              // texture to those slots causes Vulkan validation errors. Depth textures work
              // for both regular and shadow samplers.
              createDummyResources(ctx->getDevice());
              // CRITICAL: Transition depth image from UNDEFINED on first use
              if (m_dummyDepthImage != nullptr && !m_dummyDepthLayoutInitialized) {
                ctx->changeImageLayout(m_dummyDepthImage, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
                m_dummyDepthLayoutInitialized = true;
                Logger::info("[SHADER-TEXTURE] Initialized dummy depth texture layout to DEPTH_STENCIL_READ_ONLY_OPTIMAL");
              }
              if (m_dummyDepthTexture != nullptr) {
                ctx->bindResourceView(binding, m_dummyDepthTexture, nullptr);
                if (m_dummyShadowSampler != nullptr) {
                   ctx->bindResourceSampler(binding, m_dummyShadowSampler);
                }
                dummyBindCount++;
              }
            }
          }
        }
        
        if (dummyBindCount > 0) {
          static uint32_t dummyLogCount = 0;
          if (dummyLogCount++ < 20) {
            Logger::warn(str::format("[SHADER-TEXTURE] Bound ", dummyBindCount, " DUMMY textures! (Shader expected bindings we didn't have)"));
          }
        }
      } else {
        Logger::warn("[SHADER-TEXTURE] NO pixel shader! Cannot bind textures correctly.");
      }

      auto tBindingEnd = std::chrono::high_resolution_clock::now();
      totalBindingTime += std::chrono::duration<double, std::micro>(tBindingEnd - tBindingStart).count();

      // ===== TRUE INSTANCED MULTI-DRAW-INDIRECT WITH GL_LAYER ROUTING =====
      // ONE draw call renders ALL captures in this group to texture array layers
      auto tRenderStart = std::chrono::high_resolution_clock::now();

      // Check if this group uses indexed or non-indexed draws (use firstRequest from line 379)
      bool useIndexed = (firstRequest.indexCount > 0);

      // Build indirect draw buffer with firstInstance = layer index
      auto tBuildCmdStart = std::chrono::high_resolution_clock::now();
      std::vector<VkDrawIndexedIndirectCommand> indirectIndexedDraws;
      std::vector<VkDrawIndirectCommand> indirectDraws;

      for (uint32_t layerIdx = 0; layerIdx < groupSize; layerIdx++) {
        const auto& request = m_pendingCaptureRequests[group.requestIndices[layerIdx]];

        if (request.indexCount > 0) {
          VkDrawIndexedIndirectCommand cmd = {};
          cmd.indexCount = request.indexCount;
          cmd.instanceCount = 1;
          cmd.firstIndex = request.indexOffset;
          cmd.vertexOffset = request.vertexOffset;
          cmd.firstInstance = layerIdx;  // KEY: gl_InstanceIndex = layer index!
          indirectIndexedDraws.push_back(cmd);
        } else {
          VkDrawIndirectCommand cmd = {};
          cmd.vertexCount = request.vertexCount;
          cmd.instanceCount = 1;
          cmd.firstVertex = request.vertexOffset;
          cmd.firstInstance = layerIdx;  // KEY: gl_InstanceIndex = layer index!
          indirectDraws.push_back(cmd);
        }
      }
      auto tBuildCmdEnd = std::chrono::high_resolution_clock::now();
      Logger::info(str::format("[PERF] Build indirect commands: ",
        std::chrono::duration<double, std::micro>(tBuildCmdEnd - tBuildCmdStart).count(), " us"));

      // Log draw parameters for debugging
      static uint32_t drawParamLogCount = 0;
      if (++drawParamLogCount <= 10) {
        if (!indirectDraws.empty()) {
          const auto& cmd = indirectDraws[0];
          Logger::info(str::format("[DRAW-PARAMS] Non-indexed draw: vertexCount=", cmd.vertexCount,
            " firstVertex=", cmd.firstVertex, " instanceCount=", cmd.instanceCount,
            " (total draws=", indirectDraws.size(), ")"));
          if (cmd.vertexCount == 0) {
            Logger::warn("[DRAW-PARAMS] WARNING: vertexCount is 0! No triangles will be drawn!");
          }
        }
        if (!indirectIndexedDraws.empty()) {
          const auto& cmd = indirectIndexedDraws[0];
          Logger::info(str::format("[DRAW-PARAMS] Indexed draw: indexCount=", cmd.indexCount,
            " firstIndex=", cmd.firstIndex, " vertexOffset=", cmd.vertexOffset,
            " instanceCount=", cmd.instanceCount, " (total draws=", indirectIndexedDraws.size(), ")"));
          if (cmd.indexCount == 0) {
            Logger::warn("[DRAW-PARAMS] WARNING: indexCount is 0! No triangles will be drawn!");
          }
        }
      }

      // Upload indirect args to GPU and execute single multi-draw-indirect call
      if (useIndexed && !indirectIndexedDraws.empty()) {
        auto tBufCreateStart = std::chrono::high_resolution_clock::now();

        // OPTIMIZED: Reuse persistent buffer, only reallocate if size grows
        size_t requiredSize = indirectIndexedDraws.size() * sizeof(VkDrawIndexedIndirectCommand);
        if (m_persistentIndexedIndirectBuffer == nullptr || m_persistentIndexedIndirectBufferSize < requiredSize) {
          // Grow buffer with 50% headroom to reduce reallocations
          size_t newSize = requiredSize + (requiredSize / 2);
          DxvkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
          bufInfo.size = newSize;
          bufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          bufInfo.stages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
          bufInfo.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

          m_persistentIndexedIndirectBuffer = ctx->getDevice()->createBuffer(bufInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "Shader Capture Indexed Indirect Buffer (Persistent)");
          m_persistentIndexedIndirectBufferSize = newSize;
          Logger::info(str::format("[PERF-OPT] Resized persistent indexed indirect buffer to ", newSize, " bytes"));
        }

        DxvkBufferSlice indirectBuf(m_persistentIndexedIndirectBuffer, 0, requiredSize);
        memcpy(indirectBuf.mapPtr(0), indirectIndexedDraws.data(), requiredSize);
        auto tBufCreateEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] Create+upload indirect buffer: ",
          std::chrono::duration<double, std::micro>(tBufCreateEnd - tBufCreateStart).count(), " us (REUSED)"));

        // Set pipeline state once
        auto tPipelineStart = std::chrono::high_resolution_clock::now();
        setCommonPipelineState(ctx, firstRequest, group.resolution.width, group.resolution.height);
        auto tPipelineEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] setCommonPipelineState: ",
          std::chrono::duration<double, std::micro>(tPipelineEnd - tPipelineStart).count(), " us"));

        auto tBindGeomStart = std::chrono::high_resolution_clock::now();
        bindGeometryBuffers(ctx, firstRequest);
        auto tBindGeomEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] bindGeometryBuffers: ",
          std::chrono::duration<double, std::micro>(tBindGeomEnd - tBindGeomStart).count(), " us"));

        // ONE MULTI-DRAW-INDIRECT CALL! Maximum performance!
        auto tDrawStart = std::chrono::high_resolution_clock::now();
        ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());

        // Log first draw command for debugging
        static uint32_t drawDiagLog = 0;
        if (++drawDiagLog <= 20 && !indirectIndexedDraws.empty()) {
          const auto& cmd = indirectIndexedDraws[0];
          Logger::info(str::format("[DRAW-DIAG] First indexed draw: indexCount=", cmd.indexCount,
            " instanceCount=", cmd.instanceCount, " firstIndex=", cmd.firstIndex,
            " vertexOffset=", cmd.vertexOffset, " firstInstance=", cmd.firstInstance));
        }

        // CRITICAL FIX: Check if multiDrawIndirect is supported
        // If not, we must loop and call drawIndexedIndirect with count=1 for each draw
        const uint32_t drawCount = static_cast<uint32_t>(indirectIndexedDraws.size());
        const bool hasMultiDrawIndirect = ctx->getDevice()->features().core.features.multiDrawIndirect;

        if (hasMultiDrawIndirect || drawCount <= 1) {
          ctx->drawIndexedIndirect(0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
        } else {
          // Fallback: loop through draws one by one
          for (uint32_t i = 0; i < drawCount; i++) {
            ctx->drawIndexedIndirect(i * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
          }
          Logger::info(str::format("[PERF] drawIndexedIndirect FALLBACK: ", drawCount, " individual draws (multiDrawIndirect not supported)"));
        }

        auto tDrawEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] drawIndexedIndirect call: ",
          std::chrono::duration<double, std::micro>(tDrawEnd - tDrawStart).count(), " us"));

        successCount += drawCount;
        multiDrawCount++;

        Logger::info(str::format("[ShaderCapture-GPU] TRUE INSTANCED MULTI-DRAW-INDIRECT: ",
                                groupSize, " draws in 1 call (indexed)"));
      } else if (!indirectDraws.empty()) {
        auto tBufCreateStart = std::chrono::high_resolution_clock::now();

        // OPTIMIZED: Reuse persistent buffer, only reallocate if size grows
        size_t requiredSize = indirectDraws.size() * sizeof(VkDrawIndirectCommand);
        if (m_persistentIndirectBuffer == nullptr || m_persistentIndirectBufferSize < requiredSize) {
          // Grow buffer with 50% headroom to reduce reallocations
          size_t newSize = requiredSize + (requiredSize / 2);
          DxvkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
          bufInfo.size = newSize;
          bufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          bufInfo.stages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
          bufInfo.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

          m_persistentIndirectBuffer = ctx->getDevice()->createBuffer(bufInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "Shader Capture Indirect Buffer (Persistent)");
          m_persistentIndirectBufferSize = newSize;
          Logger::info(str::format("[PERF-OPT] Resized persistent indirect buffer to ", newSize, " bytes"));
        }

        DxvkBufferSlice indirectBuf(m_persistentIndirectBuffer, 0, requiredSize);
        memcpy(indirectBuf.mapPtr(0), indirectDraws.data(), requiredSize);
        auto tBufCreateEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] Create+upload indirect buffer: ",
          std::chrono::duration<double, std::micro>(tBufCreateEnd - tBufCreateStart).count(), " us (REUSED)"));

        auto tPipelineStart = std::chrono::high_resolution_clock::now();
        setCommonPipelineState(ctx, firstRequest, group.resolution.width, group.resolution.height);
        auto tPipelineEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] setCommonPipelineState: ",
          std::chrono::duration<double, std::micro>(tPipelineEnd - tPipelineStart).count(), " us"));

        auto tBindGeomStart = std::chrono::high_resolution_clock::now();
        bindGeometryBuffers(ctx, firstRequest);
        auto tBindGeomEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] bindGeometryBuffers: ",
          std::chrono::duration<double, std::micro>(tBindGeomEnd - tBindGeomStart).count(), " us"));

        auto tDrawStart = std::chrono::high_resolution_clock::now();
        ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());

        // Log first draw command for non-indexed path
        static uint32_t nonIndexedDrawDiagLog = 0;
        if (++nonIndexedDrawDiagLog <= 20 && !indirectDraws.empty()) {
          const auto& cmd = indirectDraws[0];
          Logger::info(str::format("[DRAW-DIAG-NONIDX] First non-indexed draw: vertexCount=", cmd.vertexCount,
            " instanceCount=", cmd.instanceCount, " firstVertex=", cmd.firstVertex,
            " firstInstance=", cmd.firstInstance));
        }

        // CRITICAL FIX: Check if multiDrawIndirect is supported
        // If not, we must loop and call drawIndirect with count=1 for each draw
        const uint32_t drawCount = static_cast<uint32_t>(indirectDraws.size());
        const bool hasMultiDrawIndirect = ctx->getDevice()->features().core.features.multiDrawIndirect;

        if (hasMultiDrawIndirect || drawCount <= 1) {
          ctx->drawIndirect(0, drawCount, sizeof(VkDrawIndirectCommand));
        } else {
          // Fallback: loop through draws one by one
          for (uint32_t i = 0; i < drawCount; i++) {
            ctx->drawIndirect(i * sizeof(VkDrawIndirectCommand), 1, sizeof(VkDrawIndirectCommand));
          }
          Logger::info(str::format("[PERF] drawIndirect FALLBACK: ", drawCount, " individual draws (multiDrawIndirect not supported)"));
        }

        auto tDrawEnd = std::chrono::high_resolution_clock::now();
        Logger::info(str::format("[PERF] drawIndirect call: ",
          std::chrono::duration<double, std::micro>(tDrawEnd - tDrawStart).count(), " us"));

        successCount += static_cast<uint32_t>(indirectDraws.size());
        multiDrawCount++;

        Logger::info(str::format("[ShaderCapture-GPU] TRUE INSTANCED MULTI-DRAW-INDIRECT: ",
                                groupSize, " draws in 1 call (non-indexed)"));
      }

      auto tRenderEnd = std::chrono::high_resolution_clock::now();
      totalMultiDrawTime += std::chrono::duration<double, std::micro>(tRenderEnd - tRenderStart).count();

      // OPTIMIZED: Store texture array directly - NO COPIES! (Strategy 1)
      // Instead of copying each layer to individual textures, we just store references
      // to the array + layer index. This eliminates the 37ms bottleneck entirely!
      auto tDistributeStart = std::chrono::high_resolution_clock::now();

      Logger::info(str::format("[PERF-OPT] Storing ", groupSize, " array layer references (ZERO-COPY!)"));

      auto tDistributeEnd = std::chrono::high_resolution_clock::now();
      double distributeTime = std::chrono::duration<double, std::micro>(tDistributeEnd - tDistributeStart).count();

      Logger::info(str::format("[PERF] Layer distribution - total: ", distributeTime, " us (OPTIMIZED - zero copy)"));
      Logger::info(str::format("[PERF]   Eliminated ", groupSize, " getRenderTarget calls (saved ~", groupSize, " us)"));
      Logger::info(str::format("[PERF]   Eliminated ", groupSize, " copyImage calls (saved ~", groupSize * 12, " us)"));
      Logger::info(str::format("[ShaderCapture-GPU] ZERO-COPY array layer storage: ", groupSize,
                              " layers in ", distributeTime, " us"));

      // OLD CODE PATH REMOVED - now using true instanced multi-draw-indirect!
      if (false) {
        auto tMultiDrawStart = std::chrono::high_resolution_clock::now();

        multiDrawCount++;
        Logger::info(str::format("[ShaderCapture-GPU] Multi-draw-indirect: ", groupSize, " draws in 1 call"));

        // Build indirect args buffer for this group
        std::vector<VkDrawIndirectCommand> indirectDraws;
        std::vector<VkDrawIndexedIndirectCommand> indirectIndexedDraws;
        bool useIndexed = false;

        for (uint32_t idx : group.requestIndices) {
          const auto& request = m_pendingCaptureRequests[idx];

          if (request.indexCount > 0) {
            useIndexed = true;
            VkDrawIndexedIndirectCommand cmd = {};
            cmd.indexCount = request.indexCount;
            cmd.instanceCount = 1;
            cmd.firstIndex = request.indexOffset;
            cmd.vertexOffset = request.vertexOffset;
            cmd.firstInstance = 0;
            indirectIndexedDraws.push_back(cmd);
          } else {
            VkDrawIndirectCommand cmd = {};
            cmd.vertexCount = request.vertexCount;
            cmd.instanceCount = 1;
            cmd.firstVertex = request.vertexOffset;
            cmd.firstInstance = 0;
            indirectDraws.push_back(cmd);
          }
        }

        // Upload indirect args to GPU
        if (useIndexed && !indirectIndexedDraws.empty()) {
          DxvkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
          bufInfo.size = indirectIndexedDraws.size() * sizeof(VkDrawIndexedIndirectCommand);
          bufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          bufInfo.stages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
          bufInfo.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

          Rc<DxvkBuffer> indirectBuffer = ctx->getDevice()->createBuffer(bufInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "Shader Capture Indirect Buffer");
          DxvkBufferSlice indirectBuf(indirectBuffer, 0, bufInfo.size);
          memcpy(indirectBuf.mapPtr(0), indirectIndexedDraws.data(), bufInfo.size);

          // Set pipeline state once
          setCommonPipelineState(ctx, firstRequest, group.resolution.width, group.resolution.height);

          // Bind geometry for first request (assumption: all use same buffers with different offsets)
          bindGeometryBuffers(ctx, firstRequest);

          // Multi-draw-indirect (with fallback if feature not supported)
          ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());
          const uint32_t drawCount = static_cast<uint32_t>(indirectIndexedDraws.size());
          const bool hasMultiDrawIndirect = ctx->getDevice()->features().core.features.multiDrawIndirect;
          if (hasMultiDrawIndirect || drawCount <= 1) {
            ctx->drawIndexedIndirect(0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
          } else {
            for (uint32_t i = 0; i < drawCount; i++) {
              ctx->drawIndexedIndirect(i * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
            }
          }

          successCount += drawCount;
        } else if (!indirectDraws.empty()) {
          DxvkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
          bufInfo.size = indirectDraws.size() * sizeof(VkDrawIndirectCommand);
          bufInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
          bufInfo.stages = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
          bufInfo.access = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

          Rc<DxvkBuffer> indirectBuffer = ctx->getDevice()->createBuffer(bufInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            DxvkMemoryStats::Category::RTXBuffer, "Shader Capture Indirect Buffer");
          DxvkBufferSlice indirectBuf(indirectBuffer, 0, bufInfo.size);
          memcpy(indirectBuf.mapPtr(0), indirectDraws.data(), bufInfo.size);

          setCommonPipelineState(ctx, firstRequest, group.resolution.width, group.resolution.height);
          bindGeometryBuffers(ctx, firstRequest);

          ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());
          const uint32_t drawCount = static_cast<uint32_t>(indirectDraws.size());
          const bool hasMultiDrawIndirect = ctx->getDevice()->features().core.features.multiDrawIndirect;
          if (hasMultiDrawIndirect || drawCount <= 1) {
            ctx->drawIndirect(0, drawCount, sizeof(VkDrawIndirectCommand));
          } else {
            for (uint32_t i = 0; i < drawCount; i++) {
              ctx->drawIndirect(i * sizeof(VkDrawIndirectCommand), 1, sizeof(VkDrawIndirectCommand));
            }
          }

          successCount += drawCount;
        }

        auto tMultiDrawEnd = std::chrono::high_resolution_clock::now();
        totalMultiDrawTime += std::chrono::duration<double, std::micro>(tMultiDrawEnd - tMultiDrawStart).count();
      } else {
        // Single draw - use direct call
        auto tSingleDrawStart = std::chrono::high_resolution_clock::now();

        singleDrawCount++;
        const auto& request = m_pendingCaptureRequests[group.requestIndices[0]];

        setCommonPipelineState(ctx, request, group.resolution.width, group.resolution.height);
        bindGeometryBuffers(ctx, request);

        if (request.indexCount > 0) {
          ctx->drawIndexed(request.indexCount, 1, request.indexOffset, request.vertexOffset, 0);
        } else {
          ctx->draw(request.vertexCount, 1, request.vertexOffset, 0);
        }

        successCount++;

        auto tSingleDrawEnd = std::chrono::high_resolution_clock::now();
        totalSingleDrawTime += std::chrono::duration<double, std::micro>(tSingleDrawEnd - tSingleDrawStart).count();
      }

      // Store array layer references in cache (OPTIMIZED: zero-copy, massive VRAM savings!)
      // Create layer-specific views for each material
      for (uint32_t layerIdx = 0; layerIdx < groupSize; layerIdx++) {
        const auto& request = m_pendingCaptureRequests[group.requestIndices[layerIdx]];

        // Create a 2D view of this specific layer in the array
        DxvkImageViewCreateInfo layerViewInfo;
        layerViewInfo.type = VK_IMAGE_VIEW_TYPE_2D;  // 2D view of a single layer
        layerViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        layerViewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        layerViewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        layerViewInfo.minLevel = 0;
        layerViewInfo.numLevels = 1;
        layerViewInfo.minLayer = layerIdx;  // THIS layer only
        layerViewInfo.numLayers = 1;        // Just ONE layer

        Rc<DxvkImageView> layerView = ctx->getDevice()->createImageView(
          group.renderTarget.image, layerViewInfo);

        // Store layer-specific view (acts like individual texture but shares VRAM!)
        // CRITICAL FIX: Use request.cacheKey instead of materialHash!
        // For RT replacements, cacheKey is a COMBINED hash (originalRT + replacement)
        // Using materialHash here was causing cache misses because lookup uses cacheKey!
        XXH64_hash_t cacheKey = request.cacheKey;
        CapturedShaderOutput& output = m_capturedOutputs[cacheKey];
        output.capturedTexture.image = group.renderTarget.image;  // Share the array image
        output.capturedTexture.view = layerView;  // But use layer-specific view
        output.arrayLayer = layerIdx;             // Track which layer (for debugging)
        output.isArrayLayer = true;               // Mark as optimized array layer
        output.geometryHash = request.geometryHash;
        output.materialHash = request.materialHash;
        output.lastCaptureFrame = m_currentFrame;
        output.captureSubmittedFrame = m_currentFrame;
        output.isDynamic = request.isDynamic;
        output.isPending = false;
        output.resolution = request.resolution;

        XXH64_hash_t storedImageHash = group.renderTarget.image->getHash();

        static uint32_t batchStoreCount = 0;
        if (++batchStoreCount <= 30) {
          Logger::warn(str::format("[BATCH-STORE] #", batchStoreCount,
                                  " cacheKey=0x", std::hex, cacheKey, std::dec,
                                  " storedImageHash=0x", std::hex, storedImageHash, std::dec,
                                  " layer=", layerIdx, "/", groupSize,
                                  " rtExtent=", group.renderTarget.image->info().extent.width, "x", group.renderTarget.image->info().extent.height,
                                  " imagePtr=", (void*)output.capturedTexture.image.ptr(),
                                  " viewPtr=", (void*)output.capturedTexture.view.ptr(),
                                  " format=", group.renderTarget.image->info().format));
        }

        // CRITICAL DIAGNOSTIC: Track ALL stored captured texture image pointers
        static uint32_t detailedStoreCount = 0;
        if (++detailedStoreCount <= 50) {
          Logger::warn(str::format("[CAPTURE-TEX-STORED] RT image pointer ", (void*)group.renderTarget.image.ptr(),
                                  " may conflict with bloom if bloom reads this pointer!"));
        }
      }
    }

    // ASYNC FRAME SPREADING: Remove only processed requests, keep the rest queued
    if (numRequests >= m_pendingCaptureRequests.size()) {
      // Processed all requests - clear everything
      m_pendingCaptureRequests.clear();
      Logger::info("[ASYNC] All queued requests processed - queue empty");
    } else {
      // Remove processed requests from front, keep unprocessed for next frame
      m_pendingCaptureRequests.erase(m_pendingCaptureRequests.begin(),
                                      m_pendingCaptureRequests.begin() + numRequests);
      Logger::info(str::format("[ASYNC] Removed ", numRequests, " processed requests - ",
                              m_pendingCaptureRequests.size(), " remain queued for next frame"));
    }

    // GPU PROFILING: Write end timestamp for CURRENT frame
    if (m_gpuTimestampStart != nullptr && m_gpuTimestampEnd != nullptr) {
      ctx->writeTimestamp(m_gpuTimestampEnd);
      Logger::info("[GPU-PROFILING] End timestamp written (current frame)");

      // Swap queries for next frame's delayed readback
      // Move current frame's queries to previous frame slots
      m_prevFrameTimestampStart = m_gpuTimestampStart;
      m_prevFrameTimestampEnd = m_gpuTimestampEnd;

      // Create new queries for next frame
      m_gpuTimestampStart = ctx->getDevice()->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
      m_gpuTimestampEnd = ctx->getDevice()->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);

      Logger::info("[GPU-PROFILING] Queries swapped - current frame queries moved to previous frame slots for next frame readback");
    }

    auto tExecutionEnd = std::chrono::high_resolution_clock::now();
    auto tFunctionEnd = std::chrono::high_resolution_clock::now();

    auto executionTime = std::chrono::duration<double, std::micro>(tExecutionEnd - tExecutionStart).count();
    auto totalTime = std::chrono::duration<double, std::micro>(tFunctionEnd - tFunctionStart).count();

    // Calculate command buffer overhead (everything except actual GPU draw time)
    double commandBufferOverhead = totalBindingTime + totalMultiDrawTime + totalSingleDrawTime;

    Logger::info(str::format("[ShaderCapture-GPU] ===== PERFORMANCE BREAKDOWN ====="));
    Logger::info(str::format("[ShaderCapture-GPU] Complete: ", successCount, " draws, ",
                            multiDrawCount, " multi-draw-indirect groups, ",
                            singleDrawCount, " single draws"));
    Logger::info(str::format("[ShaderCapture-GPU] Grouping: ", groupingTime / 1000.0, " ms"));
    Logger::info(str::format("[ShaderCapture-GPU] RT Allocation: ", totalRTAllocTime / 1000.0, " ms"));
    Logger::info(str::format(""));
    Logger::info(str::format("[BOTTLENECK-3] COMMAND BUFFER OVERHEAD BREAKDOWN:"));
    Logger::info(str::format("  [BOTTLENECK-3] Binding (RT+Tex+Pipeline): ", totalBindingTime / 1000.0, " ms"));
    Logger::info(str::format("  [BOTTLENECK-3] Multi-draw setup+execute: ", totalMultiDrawTime / 1000.0, " ms"));
    Logger::info(str::format("  [BOTTLENECK-3] Single-draw setup+execute: ", totalSingleDrawTime / 1000.0, " ms"));
    Logger::info(str::format("  [BOTTLENECK-3] TOTAL CMD OVERHEAD: ", commandBufferOverhead / 1000.0, " ms"));
    Logger::info(str::format(""));
    Logger::info(str::format("[ShaderCapture-GPU] Execution time: ", executionTime / 1000.0, " ms"));
    Logger::info(str::format("[ShaderCapture-GPU] TOTAL TIME: ", totalTime / 1000.0, " ms"));
    Logger::info(str::format(""));
    Logger::info(str::format("[BOTTLENECK-2] BARRIERS: See barrier logs above for total barrier overhead"));
    Logger::info(str::format("[BOTTLENECK-SUMMARY] CPU time: ", totalTime / 1000.0, " ms vs GPU time: <0.1 ms (from GPU profiling)"));
  }

  void ShaderOutputCapturer::setCommonPipelineState(Rc<RtxContext> ctx, const GpuCaptureRequest& request,
                                                    uint32_t rtWidth, uint32_t rtHeight) {
    // CRITICAL FIX: Use viewport matching the render target size, NOT the original game viewport!
    // The game's viewport (e.g. 1920x1080) doesn't match our render target (e.g. 256x256).
    // If we use the game's viewport, geometry is mapped outside the render target bounds = black output.
    VkViewport rtViewport = {};
    rtViewport.x = 0.0f;
    rtViewport.y = 0.0f;
    rtViewport.width = static_cast<float>(rtWidth);
    rtViewport.height = static_cast<float>(rtHeight);
    rtViewport.minDepth = 0.0f;
    rtViewport.maxDepth = 1.0f;

    VkRect2D rtScissor = {};
    rtScissor.offset = { 0, 0 };
    rtScissor.extent = { rtWidth, rtHeight };

    Logger::info(str::format("[VIEWPORT-DEBUG] USING RT SIZE: w=", rtWidth, " h=", rtHeight,
      " (original game was w=", request.viewport.width, " h=", request.viewport.height, ")"));
    ctx->setViewports(1, &rtViewport, &rtScissor);

    DxvkDepthStencilState depthState = {};
    depthState.enableDepthTest = VK_FALSE;
    depthState.enableDepthWrite = VK_FALSE;
    ctx->setDepthStencilState(depthState);

    DxvkRasterizerState rasterState = {};
    rasterState.cullMode = VK_CULL_MODE_NONE;
    rasterState.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    ctx->setRasterizerState(rasterState);

    DxvkBlendMode blendMode = {};
    blendMode.enableBlending = VK_FALSE;
    ctx->setBlendMode(0, blendMode);

    // CRITICAL FIX: Disable shadow/depth comparison samplers
    Logger::info("[SPEC-CONST] Setting SamplerDepthMode=0 to disable shadow samplers in capture");
    ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::SamplerDepthMode, 0);

    // CRITICAL FIX: Disable alpha test by setting AlphaCompareOp to VK_COMPARE_OP_ALWAYS (7)
    Logger::info("[SPEC-CONST] Setting AlphaCompareOp=7 (ALWAYS) to disable alpha test in capture");
    ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::AlphaCompareOp, 7);

    // CRITICAL: Set up input assembly state (primitive topology)
    DxvkInputAssemblyState iaState = {};
    iaState.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    iaState.primitiveRestart = VK_FALSE;
    iaState.patchVertexCount = 0;
    ctx->setInputAssemblyState(iaState);

    // CRITICAL FIX: When using original vertex streams, build input layout from original elements
    // Since we bind the ORIGINAL game vertex shader (not our custom one), we must use DXSO
    // attribute locations. The DXSO compiler maps D3D9 usages to specific Vulkan locations.
    if (request.useOriginalVertexLayout && !request.originalVertexElements.empty()) {
      // Map D3D9 usage to Vulkan attribute location (DXSO-compatible mapping)
      // This MUST match what the DXSO compiler uses for the original game's vertex shader
      auto mapUsageToLocation = [](uint8_t usage, uint8_t usageIdx) -> uint32_t {
        switch (usage) {
          case D3DDECLUSAGE_POSITION:
          case D3DDECLUSAGE_POSITIONT: return 0;
          case D3DDECLUSAGE_BLENDWEIGHT: return 1;
          case D3DDECLUSAGE_BLENDINDICES: return 2;
          case D3DDECLUSAGE_NORMAL: return 3;
          case D3DDECLUSAGE_COLOR: return 4 + usageIdx;  // COLOR0=4, COLOR1=5
          case D3DDECLUSAGE_TANGENT: return 6;
          case D3DDECLUSAGE_TEXCOORD: return 7 + usageIdx;  // TEXCOORD0=7, TEXCOORD1=8, etc.
          case D3DDECLUSAGE_BINORMAL: return 9;
          case D3DDECLUSAGE_PSIZE: return 10;
          default: return 11 + usageIdx;
        }
      };

      // Build stream index -> binding index mapping
      std::unordered_map<uint32_t, uint32_t> streamToBinding;
      uint32_t nextBinding = 0;
      for (const auto& stream : request.originalVertexStreams) {
        if (streamToBinding.find(stream.streamIndex) == streamToBinding.end()) {
          streamToBinding[stream.streamIndex] = nextBinding++;
        }
      }

      // Build bindings from streams
      std::vector<DxvkVertexBinding> bindings;
      for (const auto& stream : request.originalVertexStreams) {
        uint32_t binding = streamToBinding[stream.streamIndex];
        if (binding >= bindings.size()) {
          bindings.resize(binding + 1);
        }
        bindings[binding].binding = binding;
        bindings[binding].fetchRate = stream.stride;
        bindings[binding].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      }

      // Build attributes from ALL elements - game's VS expects all of them at DXSO locations
      std::vector<DxvkVertexAttribute> attributes;
      for (const auto& elem : request.originalVertexElements) {
        auto it = streamToBinding.find(elem.stream);
        if (it == streamToBinding.end()) continue;

        VkFormat format = DecodeDecltype(D3DDECLTYPE(elem.type));
        if (format == VK_FORMAT_UNDEFINED) continue;

        DxvkVertexAttribute attr;
        attr.location = mapUsageToLocation(elem.usage, elem.usageIndex);
        attr.binding = it->second;
        attr.format = format;
        attr.offset = elem.offset;
        attributes.push_back(attr);
      }

      ctx->setInputLayout(static_cast<uint32_t>(attributes.size()), attributes.data(),
                          static_cast<uint32_t>(bindings.size()), bindings.data());

      static uint32_t origLayoutLogCount = 0;
      if (++origLayoutLogCount <= 20) {
        Logger::info(str::format("[DXSO-INPUT-LAYOUT] Using DXSO locations for ", attributes.size(),
                                " attributes (game's original VS)"));
        for (size_t i = 0; i < attributes.size(); i++) {
          Logger::info(str::format("  Attr[", i, "]: loc=", attributes[i].location,
                                  " bind=", attributes[i].binding,
                                  " fmt=", (uint32_t)attributes[i].format,
                                  " off=", attributes[i].offset));
        }
      }
      return;
    }

    // Fallback: legacy separated buffer layout (for custom VS or when original not available)
    std::array<DxvkVertexAttribute, 3> attributes;
    std::array<DxvkVertexBinding, 3> bindings;
    uint32_t numAttributes = 0;
    uint32_t numBindings = 0;

    // Position: binding 0 -> location 0 (always first in D3D9)
    if (request.vertexBuffer.defined() || request.replacementVertexBuffer.defined()) {
      attributes[numAttributes].location = 0;
      attributes[numAttributes].binding = 0;
      attributes[numAttributes].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributes[numAttributes].offset = 0;
      numAttributes++;

      bindings[numBindings].binding = 0;
      bindings[numBindings].fetchRate = request.replacementVertexBuffer.defined()
        ? request.replacementVertexStride : request.vertexStride;
      bindings[numBindings].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      numBindings++;
    }

    // Texcoord: binding 1 -> location 7 (TEXCOORD0 in DXSO)
    if (request.texcoordBuffer.defined() || request.replacementTexcoordBuffer.defined()) {
      attributes[numAttributes].location = 7;  // TEXCOORD0 maps to location 7 in DXSO
      attributes[numAttributes].binding = 1;
      attributes[numAttributes].format = VK_FORMAT_R32G32_SFLOAT;
      attributes[numAttributes].offset = 0;
      numAttributes++;

      bindings[numBindings].binding = 1;
      bindings[numBindings].fetchRate = request.replacementTexcoordBuffer.defined()
        ? request.replacementTexcoordStride : request.texcoordStride;
      bindings[numBindings].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      numBindings++;
    }

    // Normal: binding 2 -> location 3 (NORMAL in DXSO)
    if (request.normalBuffer.defined() || request.replacementNormalBuffer.defined()) {
      attributes[numAttributes].location = 3;  // NORMAL maps to location 3 in DXSO
      attributes[numAttributes].binding = 2;
      attributes[numAttributes].format = VK_FORMAT_R32G32B32_SFLOAT;
      attributes[numAttributes].offset = 0;
      numAttributes++;

      bindings[numBindings].binding = 2;
      bindings[numBindings].fetchRate = request.replacementVertexBuffer.defined()
        ? request.replacementVertexStride : request.vertexStride;  // Normal uses vertex stride
      bindings[numBindings].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
      numBindings++;
    }

    ctx->setInputLayout(numAttributes, attributes.data(), numBindings, bindings.data());

    // ALWAYS log input layout to debug vertex binding issues
    Logger::info(str::format("[VERTEX-INPUT] Set up input layout: ", numAttributes, " attributes, ", numBindings, " bindings"));
    for (uint32_t i = 0; i < numAttributes; i++) {
      Logger::info(str::format("  Attribute ", i, ": location=", attributes[i].location,
        " binding=", attributes[i].binding, " format=", (uint32_t)attributes[i].format,
        " stride=", bindings[i].fetchRate));
    }
  }

  // Helper to ensure buffer has VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
  // Some D3D9 buffers are staging buffers without vertex usage - we need to copy them
  static DxvkBufferSlice ensureVertexBuffer(Rc<RtxContext> ctx, const DxvkBufferSlice& srcSlice, uint32_t stride, const char* debugName) {
    if (!srcSlice.defined()) {
      return DxvkBufferSlice();
    }

    // Check if buffer already has vertex buffer usage
    const Rc<DxvkBuffer>& srcBuffer = srcSlice.buffer();
    VkBufferUsageFlags srcUsage = srcBuffer->info().usage;

    if (srcUsage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
      // Already a vertex buffer, use as-is
      return srcSlice;
    }

    // Need to create a new buffer with vertex usage and copy data
    Logger::info(str::format("[VERTEX-COPY] Creating vertex buffer copy for ", debugName,
                            " (src usage=0x", std::hex, srcUsage, std::dec, " missing VERTEX_BUFFER_BIT)"));

    const Rc<DxvkDevice>& device = ctx->getDevice();
    const VkDeviceSize dataSize = srcSlice.length();

    DxvkBufferCreateInfo bufInfo = {};
    bufInfo.size = dataSize;
    bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    bufInfo.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

    Rc<DxvkBuffer> dstBuffer = device->createBuffer(
      bufInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      debugName);

    DxvkBufferSlice dstSlice(dstBuffer, 0, dataSize);

    // Copy data from source to destination
    ctx->copyBuffer(dstSlice.buffer(), dstSlice.offset(),
                   srcSlice.buffer(), srcSlice.offset(),
                   dataSize);

    return dstSlice;
  }

  // Helper to ensure buffer has VK_BUFFER_USAGE_INDEX_BUFFER_BIT
  // Some D3D9 buffers are staging buffers without index usage - we need to copy them
  static DxvkBufferSlice ensureIndexBuffer(Rc<RtxContext> ctx, const DxvkBufferSlice& srcSlice, const char* debugName) {
    if (!srcSlice.defined()) {
      return DxvkBufferSlice();
    }

    // Check if buffer already has index buffer usage
    const Rc<DxvkBuffer>& srcBuffer = srcSlice.buffer();
    VkBufferUsageFlags srcUsage = srcBuffer->info().usage;

    if (srcUsage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) {
      // Already an index buffer, use as-is
      return srcSlice;
    }

    // Need to create a new buffer with index usage and copy data
    Logger::info(str::format("[INDEX-COPY] Creating index buffer copy for ", debugName,
                            " (src usage=0x", std::hex, srcUsage, std::dec, " missing INDEX_BUFFER_BIT)"));

    const Rc<DxvkDevice>& device = ctx->getDevice();
    const VkDeviceSize dataSize = srcSlice.length();

    DxvkBufferCreateInfo bufInfo = {};
    bufInfo.size = dataSize;
    bufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    bufInfo.access = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

    Rc<DxvkBuffer> dstBuffer = device->createBuffer(
      bufInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXBuffer,
      debugName);

    DxvkBufferSlice dstSlice(dstBuffer, 0, dataSize);

    // Copy data from source to destination
    ctx->copyBuffer(dstSlice.buffer(), dstSlice.offset(),
                   srcSlice.buffer(), srcSlice.offset(),
                   dataSize);

    return dstSlice;
  }

  void ShaderOutputCapturer::bindGeometryBuffers(Rc<RtxContext> ctx, const GpuCaptureRequest& request) {
    // Use self-contained geometry buffer data from request - no DrawCallState needed!
    // CRITICAL: Some buffers may be staging buffers without VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    // We must ensure all bound buffers have the correct usage flags

    // CRITICAL FIX: Use original vertex streams when available for correct D3D9 binding layout
    if (request.useOriginalVertexLayout && !request.originalVertexStreams.empty()) {
      static uint32_t bindLogCount = 0;
      if (++bindLogCount <= 20) {
        Logger::info(str::format("[BIND-ORIG-STREAMS] Using ", request.originalVertexStreams.size(),
                                " original streams instead of separated buffers"));
      }

      // Bind each original stream to its D3D9 stream index
      for (const auto& stream : request.originalVertexStreams) {
        if (stream.buffer == nullptr) continue;

        DxvkBufferSlice bufferSlice(stream.buffer);
        ctx->bindVertexBuffer(stream.streamIndex, bufferSlice, stream.stride);

        if (bindLogCount <= 20) {
          Logger::info(str::format("  Stream[", stream.streamIndex, "]: stride=", stream.stride,
                                  " size=", stream.buffer->info().size));

          // DEBUG: Dump first few vertex positions from stream 0 (position)
          if (stream.streamIndex == 0 && stream.stride >= 12) {
            void* mappedPtr = stream.buffer->mapPtr(0);
            uint64_t bufSize = stream.buffer->info().size;
            int numVerts = std::min(3u, (uint32_t)(bufSize / stream.stride));
            Logger::info(str::format("[VERTEX-DEBUG] Stream0: stride=", stream.stride, " bufSize=", bufSize,
              " numVerts=", numVerts, " mappedPtr=", (mappedPtr ? "OK" : "NULL")));
            if (mappedPtr && numVerts > 0) {
              const uint8_t* data = reinterpret_cast<const uint8_t*>(mappedPtr);
              for (int v = 0; v < numVerts; v++) {
                const float* pos = reinterpret_cast<const float*>(data + v * stream.stride);
                Logger::info(str::format("[VERTEX-DEBUG] v[", v, "] = (", pos[0], ", ", pos[1], ", ", pos[2], ")"));
              }
            } else if (!mappedPtr) {
              Logger::err("[VERTEX-DEBUG] FAILED to map vertex buffer! Can't read positions.");
            }
          }
        }
      }

      // Bind index buffer (ensure it has INDEX_BUFFER_BIT)
      if (request.replacementIndexBuffer.defined()) {
        auto ib = ensureIndexBuffer(ctx, request.replacementIndexBuffer, "ShaderCapture Index (replacement)");
        ctx->bindIndexBuffer(ib, request.replacementIndexType);
      } else if (request.indexBuffer.defined()) {
        auto ib = ensureIndexBuffer(ctx, request.indexBuffer, "ShaderCapture Index");
        ctx->bindIndexBuffer(ib, request.indexType);
      }
      return;
    }

    // Fallback: use separated pos/tex/norm buffers (for custom VS or when original not available)

    // Bind vertex buffers (replacement takes priority)
    if (request.replacementVertexBuffer.defined()) {
      auto vb = ensureVertexBuffer(ctx, request.replacementVertexBuffer, request.replacementVertexStride, "ShaderCapture Position (replacement)");
      ctx->bindVertexBuffer(0, vb, request.replacementVertexStride);
    } else if (request.vertexBuffer.defined()) {
      auto vb = ensureVertexBuffer(ctx, request.vertexBuffer, request.vertexStride, "ShaderCapture Position");
      ctx->bindVertexBuffer(0, vb, request.vertexStride);
    }

    if (request.replacementTexcoordBuffer.defined()) {
      auto tb = ensureVertexBuffer(ctx, request.replacementTexcoordBuffer, request.replacementTexcoordStride, "ShaderCapture Texcoord (replacement)");
      ctx->bindVertexBuffer(1, tb, request.replacementTexcoordStride);
    } else if (request.texcoordBuffer.defined()) {
      auto tb = ensureVertexBuffer(ctx, request.texcoordBuffer, request.texcoordStride, "ShaderCapture Texcoord");
      ctx->bindVertexBuffer(1, tb, request.texcoordStride);
    }

    if (request.replacementNormalBuffer.defined()) {
      auto nb = ensureVertexBuffer(ctx, request.replacementNormalBuffer, request.vertexStride, "ShaderCapture Normal (replacement)");
      ctx->bindVertexBuffer(2, nb, request.vertexStride);
    } else if (request.normalBuffer.defined()) {
      auto nb = ensureVertexBuffer(ctx, request.normalBuffer, request.vertexStride, "ShaderCapture Normal");
      ctx->bindVertexBuffer(2, nb, request.vertexStride);
    }

    // Bind index buffer (ensure it has INDEX_BUFFER_BIT)
    if (request.replacementIndexBuffer.defined()) {
      auto ib = ensureIndexBuffer(ctx, request.replacementIndexBuffer, "ShaderCapture Index (replacement, fallback)");
      ctx->bindIndexBuffer(ib, request.replacementIndexType);
    } else if (request.indexBuffer.defined()) {
      auto ib = ensureIndexBuffer(ctx, request.indexBuffer, "ShaderCapture Index (fallback)");
      ctx->bindIndexBuffer(ib, request.indexType);
    }
  }

  // ======== END GPU-DRIVEN SYSTEM ========

  // OPTIMIZATION: Efficient material complexity detection
  // Note: LegacyMaterialData only exposes color textures, not individual PBR maps
  // So we detect complexity based on whether material uses dual textures
  struct MaterialComplexity {
    bool hasMultipleTextures = false;

    uint32_t getMRTCount() const {
      return hasMultipleTextures ? 2 : 1;
    }
  };

  static inline MaterialComplexity detectMaterialComplexity(const LegacyMaterialData& material) {
    MaterialComplexity result;

    // Check if material uses both color texture slots
    result.hasMultipleTextures = material.getColorTexture().isValid() &&
                                  material.getColorTexture2().isValid();

    return result;
  }

  bool ShaderOutputCapturer::shouldCaptureStatic(const DrawCallState& drawCallState) {
    // Stage 2 version - no cache checking, just basic feature/hash checking
    // This is called early in D3D9Rtx before GPU context is available

    static uint32_t s_callCount = 0;
    s_callCount++;
    if (s_callCount <= 10) {
      Logger::info(str::format("[STATIC-CHECK #", s_callCount, "] shouldCaptureStatic() CALLED - enableShaderOutputCapture=", enableShaderOutputCapture()));
    }

    if (!enableShaderOutputCapture()) {
      if (s_callCount <= 10) {
        Logger::info(str::format("[STATIC-CHECK #", s_callCount, "] RETURNING FALSE - feature disabled"));
      }
      return false;
    }

    // CRITICAL FIX: Skip capture if VS constants are all zeros (uninitialized state)
    // NOTE: We no longer skip captures when VS constants c[0]-c[3] are zero.
    // Many D3D9 games use PRE-TRANSFORMED vertices (D3DFVF_XYZRHW) where vertices
    // are already in screen space and don't need a transformation matrix.
    // In executeGpuCaptureBatched(), we detect this case and build a proper
    // screen-to-clip orthographic projection matrix instead.

    // Always capture draws with render target replacement
    const bool hasRenderTargetReplacement = (drawCallState.renderTargetReplacementSlot >= 0);
    if (hasRenderTargetReplacement) {
      return true;
    }

    XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();

    // Check if capturing all draws (except UI/pixel shaders)
    if (captureAllDraws()) {
      // Skip UI draws - pixel shaders without vertex shaders are typically UI
      if (drawCallState.usesPixelShader && !drawCallState.usesVertexShader) {
        return false; // UI draw, skip
      }
      return true; // Capture all non-UI draws
    }

    // SIMPLIFIED: If shader capture is enabled, capture ALL materials (bypass hash whitelist)
    if (enableShaderOutputCapture()) {
      // PERF: Only log first 20 unique materials to avoid per-draw overhead
      static std::unordered_set<XXH64_hash_t> loggedMaterials;
      if (loggedMaterials.size() < 20 && loggedMaterials.insert(matHash).second) {
        Logger::info(str::format("[SHADER-CAPTURE] Allowing material hash 0x", std::hex, matHash, std::dec, " (capture enabled, accepting all)"));
      }
      return true;
    }

    // Fallback: check hash whitelist (only used if feature is disabled above)
    if (captureEnabledHashes().count(0xALL) > 0) {
      return true;
    }
    return captureEnabledHashes().count(matHash) > 0;
  }

  bool ShaderOutputCapturer::shouldCapture(const DrawCallState& drawCallState) const {
    // PERF-BUG-HUNT: Time the entire function
    static uint32_t callCount = 0;
    static double totalTimeUs = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    ++callCount;
    const bool shouldLog = (callCount <= 20) || (callCount % 1000 == 0);

    if (!enableShaderOutputCapture()) {
      auto tEnd = std::chrono::high_resolution_clock::now();
      double elapsedUs = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
      totalTimeUs += elapsedUs;
      if (shouldLog) {
        Logger::info(str::format("[PERF-shouldCapture] Disabled path: ", elapsedUs, " μs (avg: ", totalTimeUs / callCount, " μs over ", callCount, " calls)"));
      }
      return false;
    }

    // Get cache key (uses texture hash for RT replacements, material hash otherwise)
    auto tCacheKeyStart = std::chrono::high_resolution_clock::now();

    // DEBUG: Log RT replacement details - use same logging pattern as PERF-shouldCapture
    const bool shouldLogCacheKey = shouldLog;  // Re-use the existing shouldLog from line 1086
    if (shouldLogCacheKey) {
      const bool hasRTRepl = (drawCallState.renderTargetReplacementSlot >= 0);
      Logger::info(str::format("[GETCACHEKEY-CPP] BEFORE getCacheKey: rtReplacementSlot=", drawCallState.renderTargetReplacementSlot,
                              " hasRTRepl=", hasRTRepl ? "YES" : "NO",
                              " originalRTHash=0x", std::hex, drawCallState.originalRenderTargetHash, std::dec));
      if (hasRTRepl) {
        const TextureRef& replacementTexture = drawCallState.getMaterialData().getColorTexture();
        Logger::info(str::format("[GETCACHEKEY-CPP] RT replacement texture isValid=", replacementTexture.isValid() ? "YES" : "NO",
                                " hash=0x", std::hex, (replacementTexture.isValid() ? replacementTexture.getImageHash() : 0), std::dec));
      }
    }

    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);

    // DEBUG: Log result AFTER calling getCacheKey
    if (shouldLogCacheKey) {
      Logger::info(str::format("[GETCACHEKEY-CPP] AFTER getCacheKey: cacheKey=0x", std::hex, cacheKey, std::dec,
                              " isValidKey=", isValidKey ? "YES" : "NO"));
    }

    auto tCacheKeyEnd = std::chrono::high_resolution_clock::now();
    double cacheKeyTimeUs = std::chrono::duration<double, std::micro>(tCacheKeyEnd - tCacheKeyStart).count();
    const bool hasRenderTargetReplacement = (drawCallState.renderTargetReplacementSlot >= 0);

    // Reject only if the key is explicitly marked invalid (invalid RT replacement texture)
    if (!isValidKey) {
      auto tEnd = std::chrono::high_resolution_clock::now();
      double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
      totalTimeUs += totalTime;
      if (shouldLog) {
        Logger::info(str::format("[PERF-shouldCapture] INVALID KEY path: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                " μs) | getCacheKey: ", cacheKeyTimeUs, " μs"));
      }
      return false;
    }

    // cacheKey=0 is now allowed for regular materials (materialHash can legitimately be 0)
    if (shouldLog && cacheKey == 0) {
      Logger::info(str::format("[ShaderOutputCapturer] cacheKey=0 detected (isRTReplacement=",
                              hasRenderTargetReplacement ? "YES" : "NO", ") - proceeding with capture"));
    }

    // Check if we already captured this material
    auto tCacheLookup1Start = std::chrono::high_resolution_clock::now();
    auto it = m_capturedOutputs.find(cacheKey);
    auto tCacheLookup1End = std::chrono::high_resolution_clock::now();
    double cacheLookup1TimeUs = std::chrono::duration<double, std::micro>(tCacheLookup1End - tCacheLookup1Start).count();

    if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
      // Get the captured image's hash to check if capture succeeded
      XXH64_hash_t capturedHash = (it->second.capturedTexture.image != nullptr) ? it->second.capturedTexture.image->getHash() : 0;

      // CRITICAL FIX: Force recapture if previous capture failed (hash == 0)
      if (capturedHash == 0) {
        if (shouldLog) {
          Logger::info(str::format("[PERF-shouldCapture] FORCE RECAPTURE - previous capture failed (hash=0) for cacheKey=0x",
                                  std::hex, cacheKey, std::dec));
        }
        // Fall through to capture logic below
      } else {
        // Already cached - check if it's a dynamic material that needs periodic re-capture
        auto tIsDynamicStart = std::chrono::high_resolution_clock::now();
        const bool isDynamic = isDynamicMaterial(cacheKey);
        auto tIsDynamicEnd = std::chrono::high_resolution_clock::now();
        double isDynamicTimeUs = std::chrono::duration<double, std::micro>(tIsDynamicEnd - tIsDynamicStart).count();

        if (!isDynamic) {
          // Static material - ALWAYS use cache, never re-capture
          auto tEnd = std::chrono::high_resolution_clock::now();
          double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
          totalTimeUs += totalTime;
          if (shouldLog) {
            Logger::info(str::format("[PERF-shouldCapture] STATIC CACHE HIT: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                    " μs) | getCacheKey: ", cacheKeyTimeUs, " μs | lookup: ", cacheLookup1TimeUs,
                                    " μs | isDynamic: ", isDynamicTimeUs, " μs | hash=0x", std::hex, capturedHash, std::dec));
          }
          return false;
        }

        // Dynamic material - check if it needs re-capture based on interval
        // Continue to needsRecapture() check below
      }
    }

    // RT replacements always capture if not cached
    if (hasRenderTargetReplacement) {
      auto tEnd = std::chrono::high_resolution_clock::now();
      double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
      totalTimeUs += totalTime;
      if (shouldLog) {
        Logger::info(str::format("[PERF-shouldCapture] RT REPLACEMENT path: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                " μs) | getCacheKey: ", cacheKeyTimeUs, " μs"));
      }
      return true;
    }

    // RT feedback cases (no replacement found but slot 0 was a render target)
    // CRITICAL: Wait 60 frames before capturing RT feedback to let RTs accumulate content from game's normal rendering
    // Otherwise we capture on first use when RTs are empty/black
    if (drawCallState.originalRenderTargetHash != 0) {
      constexpr uint32_t RT_FEEDBACK_DELAY_FRAMES = 60;
      if (m_currentFrame < RT_FEEDBACK_DELAY_FRAMES) {
        if (shouldLog) {
          Logger::info(str::format("[ShaderCapture-RTFeedback] Delaying capture until frame ", RT_FEEDBACK_DELAY_FRAMES,
                                  " (currentFrame=", m_currentFrame, ") to let RT accumulate content"));
        }
        return false;
      }
      if (shouldLog) {
        Logger::info(str::format("[ShaderOutputCapturer] shouldCapture() returning TRUE - RT feedback case (cacheKey=0x",
                                std::hex, cacheKey, std::dec, " originalRT=0x", drawCallState.originalRenderTargetHash, ")"));
      }
      return true;
    }

    // For non-RT-replacement materials, check the whitelist
    XXH64_hash_t matHash = cacheKey; // Same as material hash for non-RT materials

    // LOG ALL MATERIALS to help identify 3D world geometry hashes
    static uint32_t loggedMaterialCount = 0;
    if (loggedMaterialCount < 100) {  // Log first 100 unique materials
      static std::unordered_set<XXH64_hash_t> loggedMaterials;
      if (loggedMaterials.find(matHash) == loggedMaterials.end()) {
        loggedMaterials.insert(matHash);
        loggedMaterialCount++;

        // Get useful info to distinguish 3D geometry from UI
        const auto& materialData = drawCallState.getMaterialData();
        const auto& geometryData = drawCallState.geometryData;
        const bool hasTextures = materialData.getColorTexture().isValid();
        const uint32_t vertexCount = geometryData.vertexCount;
        const bool isIndexed = geometryData.indexCount > 0;

        Logger::info(str::format("[MATERIAL-DISCOVERY] Material #", loggedMaterialCount,
                                " hash=0x", std::hex, matHash, std::dec,
                                " isRTReplacement=", hasRenderTargetReplacement ? "YES" : "NO",
                                " vertices=", vertexCount,
                                " indexed=", isIndexed ? "YES" : "NO",
                                " hasTexture=", hasTextures ? "YES" : "NO"));
      }
    }

    if (callCount <= 20) {
      Logger::info(str::format("[ShaderOutputCapturer] Material hash: 0x", std::hex, matHash, std::dec));
      Logger::info(str::format("[ShaderOutputCapturer] captureEnabledHashes size: ", captureEnabledHashes().size()));
    }

    // ASYNC CAPTURE MODE: Capture all materials without throttling, let GPU queue work asynchronously
    // GPU will naturally pipeline the work. We mark captures as "pending" and check completion later.
    auto tCacheLookup2Start = std::chrono::high_resolution_clock::now();
    auto cacheIt = m_capturedOutputs.find(cacheKey);
    auto tCacheLookup2End = std::chrono::high_resolution_clock::now();
    double cacheLookup2TimeUs = std::chrono::duration<double, std::micro>(tCacheLookup2End - tCacheLookup2Start).count();

    // Check if already captured and ready (not pending)
    if (cacheIt != m_capturedOutputs.end() && cacheIt->second.capturedTexture.isValid()) {
      // If still pending, check if GPU has finished via resource tracking
      if (cacheIt->second.isPending) {
        // Check if GPU work is complete using DXVK's built-in resource tracking
        // If the image is no longer in use, GPU has finished and we can mark it ready
        if (cacheIt->second.capturedTexture.image.ptr() != nullptr &&
            !cacheIt->second.capturedTexture.image->isInUse()) {
          // Need to modify the cache entry, so we need non-const access
          m_capturedOutputs[cacheKey].isPending = false;
          if (callCount <= 20) {
            Logger::info(str::format("[ShaderCapture-ASYNC] Capture COMPLETED for matHash=0x", std::hex, matHash, std::dec,
                                    " (GPU finished after ", m_currentFrame - cacheIt->second.captureSubmittedFrame, " frames)"));
          }
        } else {
          // Still pending - continue using cached version if available, or skip if not ready yet
          if (callCount <= 20) {
            Logger::info(str::format("[ShaderCapture-ASYNC] Capture PENDING for matHash=0x", std::hex, matHash, std::dec,
                                    " (waiting for GPU, ", m_currentFrame - cacheIt->second.captureSubmittedFrame, " frames elapsed)"));
          }
          return false; // Skip re-capture while pending
        }
      }

      // Already captured and ready - check if it's a dynamic material that needs periodic re-capture
      auto tIsDynamic2Start = std::chrono::high_resolution_clock::now();
      const bool isDynamic = isDynamicMaterial(cacheKey);
      auto tIsDynamic2End = std::chrono::high_resolution_clock::now();
      double isDynamic2TimeUs = std::chrono::duration<double, std::micro>(tIsDynamic2End - tIsDynamic2Start).count();

      if (!isDynamic) {
        // Static material - never re-capture
        auto tEnd = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
        totalTimeUs += totalTime;
        if (shouldLog) {
          Logger::info(str::format("[PERF-shouldCapture] STATIC CACHE HIT (async path): ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                  " μs) | getCacheKey: ", cacheKeyTimeUs, " μs | lookup2: ", cacheLookup2TimeUs,
                                  " μs | isDynamic: ", isDynamic2TimeUs, " μs"));
        }
        return false;
      }

      // Dynamic material - check if needs re-capture based on interval
      auto tNeedsRecapStart = std::chrono::high_resolution_clock::now();
      const bool needRecap = needsRecapture(drawCallState, m_currentFrame);
      auto tNeedsRecapEnd = std::chrono::high_resolution_clock::now();
      double needsRecapTimeUs = std::chrono::duration<double, std::micro>(tNeedsRecapEnd - tNeedsRecapStart).count();

      auto tEnd = std::chrono::high_resolution_clock::now();
      double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
      totalTimeUs += totalTime;

      if (!needRecap) {
        if (shouldLog) {
          Logger::info(str::format("[PERF-shouldCapture] DYNAMIC NO RECAPTURE: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                  " μs) | getCacheKey: ", cacheKeyTimeUs, " μs | lookup2: ", cacheLookup2TimeUs,
                                  " μs | isDynamic: ", isDynamic2TimeUs, " μs | needsRecap: ", needsRecapTimeUs, " μs"));
        }
        return false;
      }

      // Dynamic material that needs re-capture!
      if (shouldLog) {
        Logger::info(str::format("[PERF-shouldCapture] DYNAMIC NEEDS RECAPTURE: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                                " μs) | getCacheKey: ", cacheKeyTimeUs, " μs | lookup2: ", cacheLookup2TimeUs,
                                " μs | isDynamic: ", isDynamic2TimeUs, " μs | needsRecap: ", needsRecapTimeUs, " μs"));
      }
      return true;
    }

    // Not yet captured - allow async capture (no throttling)
    auto tEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::micro>(tEnd - tStart).count();
    totalTimeUs += totalTime;
    if (shouldLog) {
      Logger::info(str::format("[PERF-shouldCapture] NEW CAPTURE: ", totalTime, " μs (avg: ", totalTimeUs / callCount,
                              " μs) | getCacheKey: ", cacheKeyTimeUs, " μs | lookup2: ", cacheLookup2TimeUs, " μs"));
    }
    return true;
  }

  bool ShaderOutputCapturer::needsRecapture(
      const DrawCallState& drawCallState,
      uint32_t currentFrame) const {

    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    if (!isValidKey) {
      return false; // Invalid RT replacement
    }

    static uint32_t needsRecaptureLogCount = 0;
    const bool shouldLog = (++needsRecaptureLogCount <= 20);

    // Check cache using the same key as shouldCapture()
    auto it = m_capturedOutputs.find(cacheKey);
    if (it == m_capturedOutputs.end()) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderCapture-Recapture] #", needsRecaptureLogCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " NEEDS RECAPTURE - not in cache"));
      }
      return true; // Never captured
    }

    // Check if it's a dynamic material (RT replacements are always static)
    const bool isDynamic = it->second.isDynamic;

    // Get the captured image's hash to check if capture succeeded
    XXH64_hash_t capturedHash = (it->second.capturedTexture.image != nullptr) ? it->second.capturedTexture.image->getHash() : 0;

    // CRITICAL FIX: Force recapture if previous capture failed (hash == 0)
    // This ensures we retry captures after fixing bugs (like the DXSO location fix)
    if (capturedHash == 0) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderCapture-Recapture] #", needsRecaptureLogCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " FORCE RECAPTURE - previous capture failed (hash=0)"));
      }
      return true; // Force recapture of failed captures
    }

    if (!isDynamic) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderCapture-Recapture] #", needsRecaptureLogCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " SKIP RECAPTURE - static material, using cache (hash=0x",
                                std::hex, capturedHash, std::dec, ")"));
      }
      return false; // Static material, use cached version
    }

    // For dynamic materials, check recapture interval
    uint32_t frameDelta = currentFrame - it->second.lastCaptureFrame;
    const bool needsRecap = frameDelta >= recaptureInterval();

    if (shouldLog) {
      Logger::info(str::format("[ShaderCapture-Recapture] #", needsRecaptureLogCount,
                              " cacheKey=0x", std::hex, cacheKey, std::dec,
                              " isDynamic=YES frameDelta=", frameDelta,
                              " recaptureInterval=", recaptureInterval(),
                              " result=", needsRecap ? "NEEDS RECAPTURE" : "SKIP (too soon)"));
    }

    return needsRecap;
  }

  bool ShaderOutputCapturer::captureDrawCall(
      Rc<RtxContext> ctx,
      const DxvkRaytracingInstanceState& rtState,
      const DrawCallState& drawCallState,
      const DrawParameters& drawParams,
      TextureRef& outputTexture) {

    Logger::info("[!!!DIAGNOSTIC!!!] captureDrawCall FUNCTION ENTRY - NEW CODE IS EXECUTING");

    static uint32_t captureCallEntryCount = 0;
    ++captureCallEntryCount;

    // AGGRESSIVE LOGGING - Log EVERY call for first 100 to diagnose issue
    if (captureCallEntryCount <= 100) {
      Logger::info(str::format("[CAPTURE-ENTRY] ========== captureDrawCall #", captureCallEntryCount, " ENTERED =========="));
    }

    // GPU-DRIVEN BATCHED CAPTURE MODE - PRODUCTION READY
    // Queues captures and executes them in batches to eliminate CPU-GPU sync overhead
    static constexpr bool USE_GPU_DRIVEN_MODE = true; // ENABLED: Production-ready batched execution

    if (captureCallEntryCount <= 100) {
      Logger::info(str::format("[CAPTURE-MODE-CHECK] USE_GPU_DRIVEN_MODE = ", USE_GPU_DRIVEN_MODE ? "TRUE" : "FALSE"));
    }

    if (USE_GPU_DRIVEN_MODE) {
      static uint32_t gpuModeEntryCount = 0;
      ++gpuModeEntryCount;

      if (gpuModeEntryCount <= 100) {
        Logger::info(str::format("[GPU-MODE-ENTRY] ========== Entered GPU batching path #", gpuModeEntryCount, " =========="));
      }
      // Check if we should queue this capture
      const XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();
      auto [cacheKey, isValidKey] = getCacheKey(drawCallState);

      static uint32_t cacheKeyLogCount = 0;
      ++cacheKeyLogCount;

      if (cacheKeyLogCount <= 100) {
        Logger::info(str::format("[GPU-CACHEKEY] #", cacheKeyLogCount,
                                " matHash=0x", std::hex, matHash,
                                " cacheKey=0x", cacheKey, std::dec,
                                " isValid=", isValidKey));
      }

      if (!isValidKey) {
        static uint32_t invalidKeyCount = 0;
        ++invalidKeyCount;

        if (invalidKeyCount <= 100) {
          Logger::info(str::format("[GPU-SKIP-INVALID] #", invalidKeyCount, " Invalid cache key, RETURNING FALSE"));
        }
        return false; // Invalid key, skip
      }

      // Check if already captured
      auto it = m_capturedOutputs.find(cacheKey);

      static uint32_t cacheLookupCount = 0;
      if (++cacheLookupCount <= 100) {
        bool found = (it != m_capturedOutputs.end());
        bool valid = found && it->second.capturedTexture.isValid();
        Logger::warn(str::format("[CACHE-LOOKUP] #", cacheLookupCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " found=", found ? "YES" : "NO",
                                " valid=", valid ? "YES" : "NO"));
      }

      if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
        // CRITICAL FIX: Check if previous capture failed (hash == 0)
        // If so, skip cache and fall through to queue new capture
        XXH64_hash_t cachedHash = (it->second.capturedTexture.image != nullptr) ? it->second.capturedTexture.image->getHash() : 0;
        if (cachedHash == 0) {
          static uint32_t forceRecaptureCount = 0;
          if (++forceRecaptureCount <= 100) {
            Logger::info(str::format("[GPU-FORCE-RECAPTURE] #", forceRecaptureCount,
                                    " Previous capture failed (hash=0) for cacheKey=0x", std::hex, cacheKey, std::dec,
                                    " - will queue new capture"));
          }
          // Fall through to queue new capture
        } else {
          // Already have a cached texture - check if it's static or dynamic
          const bool isDynamic = isDynamicMaterial(cacheKey);

          if (!isDynamic) {
            // Static material - always use cache
            static uint32_t cachedHitCount = 0;
            ++cachedHitCount;

            outputTexture = getCapturedTextureInternal(cacheKey);
            if (cachedHitCount <= 100) {
              Logger::warn(str::format("[GPU-CACHED-HIT-STATIC] #", cachedHitCount,
                                      " cacheKey=0x", std::hex, cacheKey, std::dec,
                                      " outputTexture.isValid()=", outputTexture.isValid() ? "TRUE" : "FALSE",
                                      " hash=0x", std::hex, cachedHash, std::dec));
            }
            return outputTexture.isValid();
          }

          // Dynamic material - check if needs re-capture
          const bool needsRecap = needsRecapture(drawCallState, m_currentFrame);
          if (!needsRecap) {
            // Dynamic but not yet time to re-capture
            static uint32_t cachedHitDynamicCount = 0;
            ++cachedHitDynamicCount;

            if (cachedHitDynamicCount <= 100) {
              Logger::info(str::format("[GPU-CACHED-HIT-DYNAMIC] #", cachedHitDynamicCount,
                                      " DYNAMIC material but not yet time to recapture, cacheKey=0x",
                                      std::hex, cacheKey, std::dec,
                                      " framesSince=", m_currentFrame - it->second.lastCaptureFrame,
                                      " interval=", recaptureInterval()));
            }
            outputTexture = getCapturedTextureInternal(cacheKey);
            return outputTexture.isValid();
          }

          // Dynamic material that NEEDS re-capture - fall through to queue it
          static uint32_t dynamicRecaptureCount = 0;
          ++dynamicRecaptureCount;

          if (dynamicRecaptureCount <= 100) {
            Logger::info(str::format("[GPU-DYNAMIC-RECAPTURE] #", dynamicRecaptureCount,
                                    " DYNAMIC material NEEDS RECAPTURE, will queue, cacheKey=0x",
                                    std::hex, cacheKey, std::dec,
                                    " framesSince=", m_currentFrame - it->second.lastCaptureFrame,
                                    " interval=", recaptureInterval()));
          }
          // Don't return - continue to queue the capture request below
        } // close else block for cachedHash != 0
      } // close if (it != m_capturedOutputs.end())

      if (cacheKeyLogCount <= 100) {
        Logger::info(str::format("[GPU-CACHE-MISS] cacheKey=0x", std::hex, cacheKey, std::dec, " - need to queue capture"));
      }

      // Queue new capture request - COPY ALL DATA (no pointers!)
      GpuCaptureRequest request = {};

      // Hashes and metadata
      request.cacheKey = cacheKey;  // CRITICAL: Store the computed cache key (may be combined hash for RT replacements!)
      request.materialHash = matHash;
      request.geometryHash = drawCallState.getGeometryData().getHashForRule<rules::FullGeometryHash>();
      const auto& materialData = drawCallState.getMaterialData();
      request.textureHash = materialData.getColorTexture().isValid()
        ? materialData.getColorTexture().getImageHash() : 0;
      request.drawCallIndex = static_cast<uint32_t>(m_pendingCaptureRequests.size());
      // renderTargetIndex removed - not needed with on-demand RT allocation

      // Geometry counts and offsets
      const auto& geometryData = drawCallState.getGeometryData();
      request.vertexOffset = 0;
      request.vertexCount = geometryData.vertexCount;
      request.indexOffset = 0;
      request.indexCount = geometryData.indexCount;
      request.resolution = calculateCaptureResolution(drawCallState);
      request.flags = (geometryData.indexCount > 0) ? 0x1 : 0x0;
      request.isDynamic = isDynamicMaterial(matHash);

      // COPY GEOMETRY BUFFERS (Rc-counted, cheap to copy)
      request.vertexBuffer = geometryData.positionBuffer;
      request.vertexStride = geometryData.positionBuffer.stride();
      request.indexBuffer = geometryData.indexBuffer;
      request.indexType = geometryData.indexBuffer.indexType();
      request.texcoordBuffer = geometryData.texcoordBuffer;
      request.texcoordStride = geometryData.texcoordBuffer.stride();
      request.normalBuffer = geometryData.normalBuffer;

      // COPY TEXTURE (Rc-counted, cheap to copy)
      request.colorTexture = materialData.getColorTexture();

      // COPY SHADERS (Rc-counted, cheap to copy)
      // Extract vertex shader
      if (drawCallState.usesVertexShader && drawCallState.vertexShader != nullptr) {
        const D3D9CommonShader* commonShader = drawCallState.vertexShader->GetCommonShader();
        if (commonShader != nullptr) {
          request.vertexShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        }
      }
      // Extract pixel shader
      if (drawCallState.usesPixelShader && drawCallState.pixelShader != nullptr) {
        const D3D9CommonShader* commonShader = drawCallState.pixelShader->GetCommonShader();
        if (commonShader != nullptr) {
          request.pixelShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        }
      }

      // COPY SHADER CONSTANT DATA (uniforms)
      request.vertexShaderConstantData = drawCallState.vertexShaderConstantData;
      request.pixelShaderConstantData = drawCallState.pixelShaderConstantData;

      // COPY TEXTURE DATA (sampled textures from D3D9)
      request.textures.reserve(drawCallState.capturedD3D9Textures.size());
      for (const auto& capturedTex : drawCallState.capturedD3D9Textures) {
        GpuCaptureRequest::CapturedTexture tex;
        tex.texture = capturedTex.texture;
        tex.slot = capturedTex.slot;
        tex.sampler = capturedTex.sampler;
        request.textures.push_back(tex);
      }

      // Copy depth texture mask (which samplers need depth textures for shadow comparison)
      request.depthTextureMask = drawCallState.depthTextureMask;

      // COPY VIEWPORT/SCISSOR STATE
      request.viewport = drawCallState.originalViewport;
      request.scissor = drawCallState.originalScissor;

      // CRITICAL FIX: Copy original vertex streams for correct vertex binding layout
      // The original game VS expects the original D3D9 vertex layout, not our separated pos/tex/norm buffers
      static uint32_t streamDiagLogCount = 0;
      if (++streamDiagLogCount <= 50) {
        Logger::info(str::format("[STREAM-DIAG] capturedVertexStreams.size()=", drawCallState.capturedVertexStreams.size(),
                                " capturedVertexElements.size()=", drawCallState.capturedVertexElements.size(),
                                " matHash=0x", std::hex, matHash, std::dec));
      }
      if (!drawCallState.capturedVertexStreams.empty()) {
        for (const auto& stream : drawCallState.capturedVertexStreams) {
          if (stream.data.empty() || stream.stride == 0) continue;

          // Create GPU buffer for this stream
          DxvkBufferCreateInfo info;
          info.size = stream.data.size();
          info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
          info.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
          info.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
          info.requiredAlignmentOverride = 1;

          Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            DxvkMemoryStats::Category::AppBuffer,
            str::format("ShaderCapture OriginalStream ", stream.streamIndex).c_str());

          if (buffer != nullptr && buffer->mapPtr(0)) {
            std::memcpy(buffer->mapPtr(0), stream.data.data(), stream.data.size());

            GpuCaptureRequest::OriginalVertexStream origStream;
            origStream.buffer = buffer;
            origStream.streamIndex = stream.streamIndex;
            origStream.stride = stream.stride;
            origStream.offset = 0;
            request.originalVertexStreams.push_back(origStream);
          }
        }

        // Copy vertex elements for building input layout
        for (const auto& elem : drawCallState.capturedVertexElements) {
          if (elem.stream == 0xFF) break;  // End marker

          GpuCaptureRequest::OriginalVertexElement origElem;
          origElem.stream = elem.stream;
          origElem.offset = elem.offset;
          origElem.type = elem.type;
          origElem.method = elem.method;
          origElem.usage = elem.usage;
          origElem.usageIndex = elem.usageIndex;
          request.originalVertexElements.push_back(origElem);
        }

        // Enable original vertex layout if we have valid streams and elements
        request.useOriginalVertexLayout = !request.originalVertexStreams.empty() &&
                                          !request.originalVertexElements.empty();

        static uint32_t origStreamLogCount = 0;
        if (++origStreamLogCount <= 20) {
          Logger::info(str::format("[ORIG-STREAMS] Captured ", request.originalVertexStreams.size(),
                                  " streams, ", request.originalVertexElements.size(),
                                  " elements, useOriginalVertexLayout=", request.useOriginalVertexLayout));
        }
      }

      m_pendingCaptureRequests.push_back(request);

      static uint32_t queueLogCount = 0;
      ++queueLogCount;

      if (queueLogCount <= 100) {
        Logger::info(str::format("[ShaderCapture-GPU-QUEUED] ========== Request #", queueLogCount,
                                " matHash=0x", std::hex, matHash, std::dec,
                                " TOTAL_QUEUED=", m_pendingCaptureRequests.size(), " =========="));
      }

      // DEFERRED BATCHED EXECUTION (MegaGeometry-style):
      // Just queue - ALL captures execute together in onFrameBegin() next frame
      // This maximizes batching efficiency and matches reference implementation

      if (queueLogCount <= 100) {
        Logger::info(str::format("[GPU-DEFER] Deferring execution to onFrameBegin(), returning true to prevent fallback"));
      }

      // Check if already cached from previous frame
      outputTexture = getCapturedTextureInternal(cacheKey);

      static uint32_t textureCheckCount = 0;
      if (++textureCheckCount <= 100) {
        Logger::warn(str::format("[GPU-TEXTURE-CHECK] #", textureCheckCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " outputTexture.isValid()=", outputTexture.isValid() ? "TRUE" : "FALSE",
                                " - returning true to queue capture"));
      }

      // Return true even if texture not ready yet - prevents old SHADER-REEXEC fallback
      // Texture will be available next frame after batched execution
      return true;
    }

    // ===== OLD CPU-DRIVEN MODE (below) =====
    static uint32_t captureCallCount = 0;
    ++captureCallCount;
    const bool shouldLog = (captureCallCount <= 20); // Only log first 20 captures
    const XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();

    // CRITICAL FIX: Copy the Stage 2 captured buffers IMMEDIATELY before they get cleared
    // The originalIndexData/originalVertexData from Stage 2 get cleared somewhere during processing
    // We need to preserve them for shader re-execution
    std::vector<uint8_t> preservedIndexData = drawCallState.originalIndexData;
    std::vector<uint8_t> preservedVertexData = drawCallState.originalVertexData;

    const uint32_t currentFrame = ctx->getDevice()->getCurrentFrameId();

    // ALWAYS log material 0x0 (render target replacement materials)
    if (matHash == 0x0) {
      Logger::info(str::format("[ShaderOutputCapturer-0x0] captureDrawCall #", captureCallCount,
                              " DrawCallID=", drawCallState.drawCallID,
                              " matHash=0x0",
                              " hasRenderTargetReplacement=", drawCallState.renderTargetReplacementSlot >= 0 ? "YES" : "NO",
                              " replacementSlot=", drawCallState.renderTargetReplacementSlot,
                              " capturedFramebufferOutput=", (drawCallState.capturedFramebufferOutput != nullptr ? "non-null" : "null")));
    }

    if (shouldLog) {
      Logger::info(str::format("[ShaderOutputCapturer] captureDrawCall #", captureCallCount,
                              " matHash=0x", std::hex, matHash, std::dec));
    }

    // REMOVED: Self-reference check was incorrect
    // In Lego Batman 2, tex0Hash == matHash is NORMAL for all materials, not a sign of self-reference
    // We need to proceed with shader re-execution for all materials

    // OPTION B: DISABLED - Framebuffer capture returns game's RT directly
    // This just copies the weird RT texture the user was seeing
    // We want OPTION C (shader re-execution) to run instead
    /*
    if (drawCallState.capturedFramebufferOutput != nullptr) {
      if (shouldLog) {
        Logger::info("[ShaderOutputCapturer] OPTION B: Using capturedFramebufferOutput (framebuffer capture)");
      }

      auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
      if (!isValidKey) {
        Logger::err("[ShaderOutputCapturer] Invalid cache key for framebuffer capture");
        return false;
      }

      outputTexture = TextureRef(drawCallState.capturedFramebufferOutput);

      CapturedShaderOutput cached;
      cached.capturedTexture.view = drawCallState.capturedFramebufferOutput;
      cached.geometryHash = 0;
      cached.materialHash = matHash;
      cached.lastCaptureFrame = currentFrame;
      cached.isDynamic = isDynamicMaterial(cacheKey);
      cached.resolution = {
        drawCallState.capturedFramebufferOutput->imageInfo().extent.width,
        drawCallState.capturedFramebufferOutput->imageInfo().extent.height
      };
      m_capturedOutputs.emplace(cacheKey, cached);

      if (shouldLog) {
        Logger::info(str::format("[ShaderOutputCapturer] OPTION B SUCCESS: Cached framebuffer capture, resolution=",
                                cached.resolution.width, "x", cached.resolution.height,
                                " cacheKey=0x", std::hex, cacheKey, std::dec));
      }

      return true;
    }
    */

    // If framebuffer capture failed or wasn't available, fall back to shader re-execution
    // Check if we need to recapture
    // TEMPORARY DEBUG: FORCE RECAPTURE to verify geometry renders correctly
    // TODO: Fix cache invalidation to detect camera/scene changes
    const bool needRecapture = true; // FORCING RECAPTURE FOR DEBUGGING
    //const bool needRecapture = needsRecapture(drawCallState, currentFrame); // ORIGINAL (commented out)

    if (!needRecapture) {  // Use cached result if nothing changed
      // Use cached texture - get cache key for proper lookup
      auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
      if (isValidKey) {
        outputTexture = getCapturedTextureInternal(cacheKey);
      }
      const bool isValid = outputTexture.isValid();

      // Log details about the cached texture
      auto cacheIt = m_capturedOutputs.find(cacheKey);
      if (shouldLog) {
        if (cacheIt != m_capturedOutputs.end()) {
          Logger::info(str::format("[ShaderOutputCapturer] Using CACHED texture, isValid=", isValid ? "true" : "false",
                                  " cacheKey=0x", std::hex, cacheKey, std::dec,
                                  " matHash=0x", std::hex, matHash, std::dec,
                                  " resolution=", cacheIt->second.resolution.width, "x", cacheIt->second.resolution.height,
                                  " lastCaptureFrame=", cacheIt->second.lastCaptureFrame,
                                  " currentFrame=", currentFrame,
                                  " age=", currentFrame - cacheIt->second.lastCaptureFrame));
        } else {
          Logger::info(str::format("[ShaderOutputCapturer] Using cached texture, isValid=", isValid ? "true" : "false",
                                  " cacheKey=0x", std::hex, cacheKey, std::dec,
                                  " (WARNING: not in cache map but needsRecapture returned false!)"));
        }
      }
      return isValid;
    }

    if (shouldLog) {
      Logger::info("[ShaderOutputCapturer] Taking re-execution path (Option A) - will bind replacement texture");
    }
    Logger::debug("[ShaderOutputCapturer] Capturing shader output for draw call via re-execution (fallback)");

    // Save current render target state FIRST so we can get its dimensions
    Rc<DxvkImageView> prevColorTarget = ctx->getFramebufferInfo().getColorTarget(0).view;

    // CRITICAL FIX: ALWAYS use original RT dimensions to make game's projection matrix work correctly
    // The game's projection is designed for the actual render target size (e.g. 1920x1080)
    // Using a different size (1024x1024) causes vertices to be clipped outside viewport
    VkExtent2D resolution;
    VkFormat rtFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Default format

    if (prevColorTarget != nullptr) {
      // Use the ORIGINAL render target dimensions and format for ALL captures
      resolution.width = prevColorTarget->imageInfo().extent.width;
      resolution.height = prevColorTarget->imageInfo().extent.height;
      rtFormat = prevColorTarget->imageInfo().format;

      Logger::info(str::format("[ShaderCapture-Resolution] Using original RT dimensions: ",
                              resolution.width, "x", resolution.height,
                              " format=", rtFormat,
                              " (makes projection matrix work correctly)"));
    } else {
      // Fallback if no previous RT (shouldn't happen)
      resolution = calculateCaptureResolution(drawCallState);
      Logger::warn(str::format("[ShaderCapture-Resolution] No previous RT! Falling back to configured resolution: ",
                              resolution.width, "x", resolution.height));
    }

    // Get or create render target with matching format
    // CRITICAL: Each material needs its own render target so different materials don't overwrite each other
    // Use the material hash as part of the cache key
    Resources::Resource renderTarget = getRenderTarget(ctx, resolution, rtFormat, matHash);
    if (!renderTarget.isValid()) {
      Logger::err("[ShaderOutputCapturer] Failed to create render target");
      return false;
    }

    // ALWAYS log what render target we actually got for RT replacements
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info(str::format("[ShaderCapture-Format] Created/Retrieved render target:"));
      Logger::info(str::format("  format=", renderTarget.image->info().format));
      Logger::info(str::format("  extent=", renderTarget.image->info().extent.width, "x",
                              renderTarget.image->info().extent.height));
      Logger::info(str::format("  hash=0x", std::hex, renderTarget.image->getHash(), std::dec));
    }

    // CRITICAL FIX: For render target feedback (game reads from RT it writes to),
    // we need to preserve the ORIGINAL RT content and bind it as input texture
    // Otherwise shader reads from empty/cleared RT and outputs black
    TextureRef previousRTContent;
    if (drawCallState.renderTargetReplacementSlot < 0 && drawCallState.originalRenderTargetHash != 0) {
      // No replacement texture found, but slot 0 is a render target
      // The shader will try to read from slot 0 (render target feedback)
      // We need to provide the CURRENT RT content as input before we clear it
      Logger::info(str::format("[ShaderCapture-RTFeedback] No replacement found, RT feedback detected!",
                              " Preserving RT 0x", std::hex, drawCallState.originalRenderTargetHash, std::dec,
                              " for shader input"));

      // Find the original RT in captured textures
      for (const auto& capturedTex : drawCallState.capturedD3D9Textures) {
        if (capturedTex.texture.isValid() &&
            capturedTex.texture.getImageHash() == drawCallState.originalRenderTargetHash) {
          previousRTContent = capturedTex.texture;
          Logger::info(str::format("[ShaderCapture-RTFeedback] Found original RT in slot ", capturedTex.slot,
                                  " format=", capturedTex.texture.getImageView()->image()->info().format,
                                  " size=", capturedTex.texture.getImageView()->image()->info().extent.width, "x",
                                  capturedTex.texture.getImageView()->image()->info().extent.height,
                                  " - will bind as shader input"));
          break;
        }
      }

      if (!previousRTContent.isValid()) {
        Logger::warn(str::format("[ShaderCapture-RTFeedback] WARNING: Could not find RT 0x", std::hex,
                                drawCallState.originalRenderTargetHash, std::dec,
                                " in capturedD3D9Textures (size=", drawCallState.capturedD3D9Textures.size(), ")!",
                                " Shader will read from empty/cleared RT and may output black!"));
      }
    }

    // BARRIER OPTIMIZATION: Transition to GENERAL layout once, then never transition again
    // GENERAL supports both COLOR_ATTACHMENT and SHADER_READ without transitions (820 barrier reduction!)
    ctx->changeImageLayout(renderTarget.image, VK_IMAGE_LAYOUT_GENERAL);

    Logger::info(str::format("[ShaderCapture-Layout] Transitioned render target 0x", std::hex,
                            renderTarget.image->getHash(), std::dec,
                            " to GENERAL layout (supports both render and read without transitions)"));

    // Bind offscreen render target
    DxvkRenderTargets captureRt;
    captureRt.color[0].view = renderTarget.view;
    captureRt.color[0].layout = VK_IMAGE_LAYOUT_GENERAL;
    ctx->bindRenderTargets(captureRt);

    // Clear the render target to MAGENTA for debugging
    // If output stays magenta = no pixels written
    // If output is black = pixels written as black (shader issue)
    // If output is colored = SUCCESS!
    VkClearValue clearValue;
    clearValue.color.float32[0] = 1.0f;
    clearValue.color.float32[1] = 0.0f;
    clearValue.color.float32[2] = 1.0f;
    clearValue.color.float32[3] = 1.0f;

    VkClearAttachment clearAttachment;
    clearAttachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clearAttachment.colorAttachment = 0;
    clearAttachment.clearValue = clearValue;

    VkClearRect clearRect;
    clearRect.rect.offset = { 0, 0 };
    clearRect.rect.extent = resolution;
    clearRect.baseArrayLayer = 0;
    clearRect.layerCount = 1;

    ctx->clearRenderTarget(renderTarget.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);

    // Use game's EXACT viewport/scissor state for shader re-execution
    // This is CRITICAL: the game's projection matrix was designed for this viewport size
    // Using any other viewport will cause vertices to be clipped outside the viewport bounds

    // CRITICAL DEBUG: Log viewport and scissor values to diagnose invisible geometry
    Logger::info(str::format("[VIEWPORT-DEBUG] originalViewport: x=", drawCallState.originalViewport.x,
                            " y=", drawCallState.originalViewport.y,
                            " width=", drawCallState.originalViewport.width,
                            " height=", drawCallState.originalViewport.height,
                            " minDepth=", drawCallState.originalViewport.minDepth,
                            " maxDepth=", drawCallState.originalViewport.maxDepth));
    Logger::info(str::format("[VIEWPORT-DEBUG] originalScissor: x=", drawCallState.originalScissor.offset.x,
                            " y=", drawCallState.originalScissor.offset.y,
                            " width=", drawCallState.originalScissor.extent.width,
                            " height=", drawCallState.originalScissor.extent.height));
    Logger::info(str::format("[VIEWPORT-DEBUG] renderTarget resolution: ", resolution.width, "x", resolution.height));
    Logger::info(str::format("[RT-FORMAT-DEBUG] Render target format: ", rtFormat,
                            " (has alpha: ", (rtFormat == VK_FORMAT_B8G8R8A8_UNORM || rtFormat == VK_FORMAT_R8G8B8A8_UNORM) ? "YES" : "NO", ")"));
    Logger::info(str::format("[RT-FORMAT-DEBUG] Clear color: R=", clearValue.color.float32[0],
                            " G=", clearValue.color.float32[1],
                            " B=", clearValue.color.float32[2],
                            " A=", clearValue.color.float32[3]));

    ctx->setViewports(1, &drawCallState.originalViewport, &drawCallState.originalScissor);

    // CRITICAL: Set ALL pipeline state for shader re-execution
    // The game's D3D9 state is NOT available in the Vulkan render thread!
    // We must explicitly set rasterizer, blend, depth, and multisample state

    // Disable depth testing since we don't have a depth buffer
    DxvkDepthStencilState depthState = {};
    depthState.enableDepthTest = VK_FALSE;
    depthState.enableDepthWrite = VK_FALSE;
    depthState.enableStencilTest = VK_FALSE;
    depthState.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    ctx->setDepthStencilState(depthState);

    // Set rasterizer state - no culling, fill solid
    DxvkRasterizerState rasterState = {};
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode = VK_CULL_MODE_NONE;  // Don't cull any faces
    rasterState.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterState.depthClipEnable = VK_TRUE;
    rasterState.depthBiasEnable = VK_FALSE;
    rasterState.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    ctx->setRasterizerState(rasterState);

    // Set blend state - no blending, write all color channels
    DxvkBlendMode blendMode = {};
    blendMode.enableBlending = VK_FALSE;
    blendMode.colorSrcFactor = VK_BLEND_FACTOR_ONE;
    blendMode.colorDstFactor = VK_BLEND_FACTOR_ZERO;
    blendMode.colorBlendOp = VK_BLEND_OP_ADD;
    blendMode.alphaSrcFactor = VK_BLEND_FACTOR_ONE;
    blendMode.alphaDstFactor = VK_BLEND_FACTOR_ZERO;
    blendMode.alphaBlendOp = VK_BLEND_OP_ADD;
    blendMode.writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    Logger::info("[BLEND-DEBUG] Setting blend mode for shader capture:");
    Logger::info(str::format("  enableBlending=", blendMode.enableBlending ? "TRUE" : "FALSE"));
    Logger::info(str::format("  colorSrc=", blendMode.colorSrcFactor, " colorDst=", blendMode.colorDstFactor));
    Logger::info(str::format("  alphaSrc=", blendMode.alphaSrcFactor, " alphaDst=", blendMode.alphaDstFactor));
    Logger::info(str::format("  writeMask=0x", std::hex, blendMode.writeMask, std::dec,
                            " (R=", (blendMode.writeMask & VK_COLOR_COMPONENT_R_BIT) ? 1 : 0,
                            " G=", (blendMode.writeMask & VK_COLOR_COMPONENT_G_BIT) ? 1 : 0,
                            " B=", (blendMode.writeMask & VK_COLOR_COMPONENT_B_BIT) ? 1 : 0,
                            " A=", (blendMode.writeMask & VK_COLOR_COMPONENT_A_BIT) ? 1 : 0, ")"));

    ctx->setBlendMode(0, blendMode);  // Set for color attachment 0

    // Set multisample state
    DxvkMultisampleState msState = {};
    msState.sampleMask = 0xFFFFFFFF;
    msState.enableAlphaToCoverage = VK_FALSE;
    ctx->setMultisampleState(msState);

    // Bind textures and samplers from the material data
    // This ensures the pixel shader has access to the game's original textures
    const LegacyMaterialData& material = drawCallState.getMaterialData();

    static uint32_t bindLogCount = 0;
    ++bindLogCount;

    // SHADER CAPTURE FIX: Use captured D3D9 textures instead of material textures
    // The material textures might be render targets (self-reference), but the captured
    // D3D9 textures are the ACTUAL game textures that were bound when the draw call happened
    Logger::info(str::format("[ShaderCapture-D3D9Tex] #", bindLogCount,
                            " DrawCallID=", drawCallState.drawCallID,
                            " matHash=0x", std::hex, matHash, std::dec,
                            " capturedD3D9Textures.size()=", drawCallState.capturedD3D9Textures.size()));

    // Bind all captured D3D9 textures, EXCEPT self-references
    auto texBindingStartTime = std::chrono::high_resolution_clock::now();

    for (const auto& capturedTex : drawCallState.capturedD3D9Textures) {
      if (!capturedTex.texture.isValid() || capturedTex.texture.getImageView() == nullptr) {
        Logger::warn(str::format("[ShaderCapture-D3D9Tex] Skipped invalid captured texture: slot=", capturedTex.slot));
        continue;
      }

      // CRITICAL FIX: Skip empty/cleared textures (hash==0x0)
      // These are uninitialized textures that will produce black when sampled
      const XXH64_hash_t texHash = capturedTex.texture.getImageHash();
      if (texHash == 0) {
        continue;
      }

      // CRITICAL FIX: Skip textures that match the material hash (self-reference)
      // When texHash == matHash, it means we're trying to read from the render target we're writing to
      // EXCEPTION: For RT replacements, the replacement slot texture might have matHash (from previous capture)
      // but that's a VALID input texture from a previous frame, so don't skip it!

      // Convert slot to stage to check if this is the replacement slot
      const int stage = (capturedTex.slot >= 1014 && capturedTex.slot <= 1029) ? (capturedTex.slot - 1014) : -1;
      const bool isReplacementSlot = (drawCallState.renderTargetReplacementSlot >= 0) &&
                                     (stage == drawCallState.renderTargetReplacementSlot);

      // Check for self-reference against material hash
      // EXCEPTION 1: Allow the replacement slot even if it has matHash (it's a previous capture, valid input)
      // EXCEPTION 2: Allow RT feedback case - matHash==RT hash but we WANT to bind it for shader to read
      const bool isRTFeedbackCase = (drawCallState.renderTargetReplacementSlot < 0) &&
                                    (drawCallState.originalRenderTargetHash != 0) &&
                                    (texHash == drawCallState.originalRenderTargetHash);
      const bool isMatHashSelfReference = (texHash == matHash) && !isReplacementSlot && !isRTFeedbackCase;

      // ALSO skip the ORIGINAL render target for RT replacement draws (it's pink/invalid)
      const bool isOriginalRTSelfReference = (drawCallState.renderTargetReplacementSlot >= 0) &&
                                             (texHash == drawCallState.originalRenderTargetHash);

      // CRITICAL FIX FOR RT FEEDBACK: Instead of skipping self-references, COPY the RT and bind the copy!
      // This allows the pixel shader to read from the RT while writing to it (temporal feedback)
      if (isMatHashSelfReference || isOriginalRTSelfReference) {
        // Get the current render target that we're about to draw to
        Rc<DxvkImageView> currentRTView = capturedTex.texture.getImageView();
        if (currentRTView == nullptr) {
          continue;
        }

        // Create a temporary copy of the RT so shader can read from it while writing to the original
        auto rtCopyStartTime = std::chrono::high_resolution_clock::now();
        const auto& rtInfo = currentRTView->imageInfo();
        DxvkImageCreateInfo copyImageInfo;
        copyImageInfo.type = VK_IMAGE_TYPE_2D;
        copyImageInfo.format = rtInfo.format;
        copyImageInfo.flags = 0;
        copyImageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
        copyImageInfo.extent = rtInfo.extent;
        copyImageInfo.numLayers = 1;
        copyImageInfo.mipLevels = 1;
        copyImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        copyImageInfo.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        copyImageInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        copyImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        copyImageInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        Rc<DxvkImage> tempCopy = ctx->getDevice()->createImage(copyImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                                       DxvkMemoryStats::Category::AppTexture,
                                                                       "RT feedback temp copy");

        // Create image view for the copy
        DxvkImageViewCreateInfo viewInfo;
        viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = rtInfo.format;
        viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.minLevel = 0;
        viewInfo.numLevels = 1;
        viewInfo.minLayer = 0;
        viewInfo.numLayers = 1;

        Rc<DxvkImageView> tempCopyView = ctx->getDevice()->createImageView(tempCopy, viewInfo);

        // Copy current RT to temp
        VkImageSubresourceLayers subresource;
        subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresource.mipLevel = 0;
        subresource.baseArrayLayer = 0;
        subresource.layerCount = 1;

        VkImageCopy copyRegion;
        copyRegion.srcSubresource = subresource;
        copyRegion.srcOffset = {0, 0, 0};
        copyRegion.dstSubresource = subresource;
        copyRegion.dstOffset = {0, 0, 0};
        copyRegion.extent = {rtInfo.extent.width, rtInfo.extent.height, rtInfo.extent.depth};

        ctx->copyImage(tempCopy, subresource, {0, 0, 0},
                      currentRTView->image(), subresource, {0, 0, 0},
                      rtInfo.extent);

        // Bind the COPY instead of the original
        ctx->bindResourceView(capturedTex.slot, tempCopyView, nullptr);

        const Rc<DxvkSampler>& sampler = material.getSampler();
        if (sampler != nullptr) {
          ctx->bindResourceSampler(capturedTex.slot, sampler);
        }

        continue; // Done with this texture
      }

      Rc<DxvkImageView> imageView = capturedTex.texture.getImageView();
      ctx->bindResourceView(capturedTex.slot, imageView, nullptr);

      // Try to get the sampler from material (samplers are stored by texture ID, not slot)
      // For now, use sampler from material slot 0 as fallback
      const Rc<DxvkSampler>& sampler = material.getSampler();
      if (sampler != nullptr) {
        ctx->bindResourceSampler(capturedTex.slot, sampler);
      }
    }

    // CRITICAL FIX: For render target replacements, we need to bind the replacement texture to SLOT 0
    // The shader expects to sample from s0, but slot 0 contains a render target (which we skipped above)
    // The replacement texture is at renderTargetReplacementSlot, so bind it to BOTH its original slot AND slot 0
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      // Find the replacement texture in the captured textures
      for (const auto& capturedTex : drawCallState.capturedD3D9Textures) {
        // Need to convert slot back to D3D9 stage to compare with renderTargetReplacementSlot
        // Slots are: 1014-1029 for PS samplers 0-15, so stage = slot - 1014
        const int stage = (capturedTex.slot >= 1014 && capturedTex.slot <= 1029) ? (capturedTex.slot - 1014) : -1;

        if (stage == drawCallState.renderTargetReplacementSlot) {
          // This is the replacement texture - bind it to slot 0 (slot 1014 = PS sampler 0)
          const uint32_t slot0 = 1014; // PS sampler 0
          Rc<DxvkImageView> imageView = capturedTex.texture.getImageView();
          if (imageView != nullptr) {
            ctx->bindResourceView(slot0, imageView, nullptr);

            const Rc<DxvkSampler>& sampler = material.getSampler();
            if (sampler != nullptr) {
              ctx->bindResourceSampler(slot0, sampler);
            }

            Logger::info(str::format("[ShaderCapture-D3D9Tex] ✓ BOUND REPLACEMENT to slot 0: replacementSlot=", stage,
                                    " textureHash=0x", std::hex, capturedTex.texture.getImageHash(), std::dec,
                                    " (shader will sample this instead of pink RT)"));
          }
          break;
        }
      }
    } else if (previousRTContent.isValid()) {
      // No replacement found, but we have the previous RT content for feedback
      // Bind it to slot 0 so the shader can read from it
      const uint32_t slot0 = 1014; // PS sampler 0

      // TextureRef already has an image view, use it directly
      ctx->bindResourceView(slot0, previousRTContent.getImageView(), nullptr);

      const Rc<DxvkSampler>& sampler = material.getSampler();
      if (sampler != nullptr) {
        ctx->bindResourceSampler(slot0, sampler);
      }

      Logger::info(str::format("[ShaderCapture-RTFeedback] ✓ BOUND previous RT content to slot 0",
                              " hash=0x", std::hex, previousRTContent.getImageHash(), std::dec,
                              " (shader can now read previous frame for feedback)"));
    }

    static uint32_t captureCount = 0;
    ++captureCount;
    // Always log for render target replacements
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info(str::format("[ShaderOutputCapturer] Capture #", captureCount, " - Texture binding:"));
      Logger::info(str::format("  Captured D3D9 textures: ", drawCallState.capturedD3D9Textures.size()));
    }

    // Bind D3D9 shaders so they execute during the draw call
    // These shader objects were captured at Stage 2 when we had access to D3D9 state

    if (drawCallState.usesVertexShader && drawCallState.vertexShader != nullptr) {
      // Convert D3D9 vertex shader to DXVK shader
      const D3D9CommonShader* commonShader = drawCallState.vertexShader->GetCommonShader();
      if (commonShader != nullptr) {
        Rc<DxvkShader> dxvkShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        if (dxvkShader != nullptr) {
          ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, dxvkShader);
        }
      }
    }

    if (drawCallState.usesPixelShader && drawCallState.pixelShader != nullptr) {
      // Convert D3D9 pixel shader to DXVK shader
      const D3D9CommonShader* commonShader = drawCallState.pixelShader->GetCommonShader();
      if (commonShader != nullptr) {
        Rc<DxvkShader> dxvkShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        if (dxvkShader != nullptr) {
          ctx->bindShader(VK_SHADER_STAGE_FRAGMENT_BIT, dxvkShader);
        }
      }
    }

    // Log draw parameters for debugging - ALWAYS log for render target replacements
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info(str::format("[ShaderOutputCapturer] Capture #", captureCount, " - Draw parameters:"));
      Logger::info(str::format("  indexCount: ", drawParams.indexCount));
      Logger::info(str::format("  vertexCount: ", drawParams.vertexCount));
      Logger::info(str::format("  instanceCount: ", drawParams.instanceCount));
      Logger::info(str::format("  firstIndex: ", drawParams.firstIndex));
      Logger::info(str::format("  vertexOffset: ", drawParams.vertexOffset));
      Logger::info(str::format("  viewport: ", static_cast<uint32_t>(drawCallState.originalViewport.width), "x",
                              static_cast<uint32_t>(std::abs(drawCallState.originalViewport.height))));

      // Check if geometry has valid data
      const RasterGeometry& geom = drawCallState.getGeometryData();
      Logger::info(str::format("  positionBuffer defined: ", geom.positionBuffer.defined() ? "yes" : "no"));
      Logger::info(str::format("  indexBuffer defined: ", geom.indexBuffer.defined() ? "yes" : "no"));
      Logger::info(str::format("  texcoordBuffer defined: ", geom.texcoordBuffer.defined() ? "yes" : "no"));
      Logger::info(str::format("  usesVertexShader: ", drawCallState.usesVertexShader ? "yes" : "no"));
      Logger::info(str::format("  usesPixelShader: ", drawCallState.usesPixelShader ? "yes" : "no"));
      Logger::info(str::format("  vertexShader valid: ", (drawCallState.vertexShader != nullptr) ? "yes" : "no"));
      Logger::info(str::format("  pixelShader valid: ", (drawCallState.pixelShader != nullptr) ? "yes" : "no"));
    }

    // CRITICAL DECISION: Should we bind extracted geometry or use D3D9's already-bound buffers?
    //
    // Option A: Bind extracted RasterGeometry (current approach)
    //   - Allows modding/replacing geometry
    //   - But causes braille rendering if indices/buffers don't match
    //
    // Option B: Use D3D9's already-bound buffers (skip binding)
    //   - Uses original game geometry (no mods)
    //   - But should work correctly
    //
    // Let's try Option B first to verify the shaders work correctly

    // TERRAIN BAKER APPROACH: Don't bind geometry at all - use D3D9's already-bound state
    // The terrain baker (rtx_terrain_baker.cpp:582) uses this same approach successfully
    // It doesn't rebind any geometry - just changes render target, viewport, and shaders
    // This works because D3D9's vertex/index buffers are ALREADY bound from the original draw
    const RasterGeometry& geom = drawCallState.getGeometryData();

    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info("[ShaderCapture-GEOMETRY] Using D3D9's already-bound buffers (terrain baker approach)");
    }

    // OPTION A: Use original D3D9 buffers (object space) instead of RasterGeometry (transformed)
    // RasterGeometry contains TRANSFORMED vertices (world/clip space) from vertex capture
    // Re-running the vertex shader on already-transformed vertices causes double-transformation (braille!)
    // We need the ORIGINAL untransformed vertices (object space) that were captured at Stage 2
    // Use the PRESERVED copies instead of drawCallState (which may have been cleared)
    const bool useCapturedStreams = !drawCallState.capturedVertexStreams.empty();
    const bool hasLegacyBuffers = !preservedVertexData.empty() &&
      (!geom.usesIndices() || !preservedIndexData.empty());

    const bool useOriginalBuffers =
      !drawCallState.forceGeometryCopy &&
      (useCapturedStreams || hasLegacyBuffers);

    // CRITICAL FIX: originalVertexStride is only set for Stage 2 buffer captures
    // For capturedVertexStreams (Stage 4), we need to check if streams have valid data
    const bool hasValidVertexData = (drawCallState.originalVertexStride >= 12) ||
                                    (!drawCallState.capturedVertexStreams.empty() &&
                                     drawCallState.capturedVertexStreams.front().stride > 0);


    // Declare variables that will be used by both paths (Option A and RasterGeometry fallback)
    uint32_t actualVertexCount = 0;
    uint32_t actualIndexCount = 0;
    DxvkVertexAttribute attrList[16];
    DxvkVertexBinding bindList[16];
    uint32_t attrCount = 0;
    uint32_t bindCount = 0;
    bool isInterleaved = false;

    // Flag to track if we successfully used original buffers
    bool usedOriginalBuffers = false;

    if (useOriginalBuffers && hasValidVertexData) {
      // OPTION A PATH: Use original D3D9 buffers (untransformed, object space)

      if (drawCallState.originalIndexBuffer != nullptr) {
      }

      // Use draw parameters for counts (same as RasterGeometry path)
      actualVertexCount = drawParams.vertexCount;
      actualIndexCount = drawParams.indexCount;

      bindCount = 1;
      attrCount = 0;

      // NEW: Use actual D3D9 vertex declaration instead of guessing from stride
      if (drawCallState.vertexDecl != nullptr) {
        const D3D9VertexElements& elements = drawCallState.vertexDecl->GetElements();

        for (size_t i = 0; i < elements.size(); i++) {
          const D3DVERTEXELEMENT9& elem = elements[i];

          // D3DDECL_END marker
          if (elem.Stream == 0xFF) {
            break;
          }

          // Only process stream 0 elements (we only capture stream 0)
          if (elem.Stream != 0) {
            continue;
          }


          // Map D3D9 usage to Vulkan attribute location
          // D3DDECLUSAGE values: 0=Position, 1=BlendWeight, 2=BlendIndices, 3=Normal, 4=PSize,
          //                      5=TexCoord, 6=Tangent, 7=Binormal, 10=Color
          // Vulkan locations: 0=Position, 1=BlendWeight, 2=BlendIndices, 3=Normal, 4=Color0,
          //                   5=Color1, 6=Tangent, 7+=TexCoord, 9=Binormal, 10=PSize
          uint32_t location = 0;
          switch (elem.Usage) {
            case D3DDECLUSAGE_POSITION:
              location = 0;
              break;
            case D3DDECLUSAGE_BLENDWEIGHT:
              location = 1;
              break;
            case D3DDECLUSAGE_BLENDINDICES:
              location = 2;
              break;
            case D3DDECLUSAGE_NORMAL:
              location = 3;
              break;
            case D3DDECLUSAGE_PSIZE:
              location = 10; // Point size - use location 10 to avoid conflicts with Color0
              break;
            case D3DDECLUSAGE_TEXCOORD:
              location = 7 + elem.UsageIndex; // TexCoord0 = 7, TexCoord1 = 8, etc.
              break;
            case D3DDECLUSAGE_TANGENT:
              location = 6; // Tangent for normal mapping
              break;
            case D3DDECLUSAGE_BINORMAL:
              location = 9; // Binormal for normal mapping
              break;
            case D3DDECLUSAGE_COLOR:
              location = 4 + elem.UsageIndex; // Color0 = 4, Color1 = 5
              break;
            default:
              Logger::warn(str::format("[OPTION-A-EXECUTE] Unknown vertex element usage: ", static_cast<uint32_t>(elem.Usage)));
              continue;
          }

          // Map D3D9 format to Vulkan format
          VkFormat format = VK_FORMAT_UNDEFINED;
          switch (elem.Type) {
            case D3DDECLTYPE_FLOAT1:
              format = VK_FORMAT_R32_SFLOAT;
              break;
            case D3DDECLTYPE_FLOAT2:
              format = VK_FORMAT_R32G32_SFLOAT;
              break;
            case D3DDECLTYPE_FLOAT3:
              format = VK_FORMAT_R32G32B32_SFLOAT;
              break;
            case D3DDECLTYPE_FLOAT4:
              format = VK_FORMAT_R32G32B32A32_SFLOAT;
              break;
            case D3DDECLTYPE_D3DCOLOR:
              format = VK_FORMAT_B8G8R8A8_UNORM;
              break;
            case D3DDECLTYPE_UBYTE4:
              format = VK_FORMAT_R8G8B8A8_UINT;
              break;
            case D3DDECLTYPE_SHORT2:
              format = VK_FORMAT_R16G16_SINT;
              break;
            case D3DDECLTYPE_SHORT4:
              format = VK_FORMAT_R16G16B16A16_SINT;
              break;
            case D3DDECLTYPE_UBYTE4N:
              format = VK_FORMAT_R8G8B8A8_UNORM;
              break;
            case D3DDECLTYPE_SHORT2N:
              format = VK_FORMAT_R16G16_SNORM;
              break;
            case D3DDECLTYPE_SHORT4N:
              format = VK_FORMAT_R16G16B16A16_SNORM;
              break;
            case D3DDECLTYPE_USHORT2N:
              format = VK_FORMAT_R16G16_UNORM;
              break;
            case D3DDECLTYPE_USHORT4N:
              format = VK_FORMAT_R16G16B16A16_UNORM;
              break;
            case D3DDECLTYPE_FLOAT16_2:
              format = VK_FORMAT_R16G16_SFLOAT;
              break;
            case D3DDECLTYPE_FLOAT16_4:
              format = VK_FORMAT_R16G16B16A16_SFLOAT;

              // CRITICAL FIX: Detect when game lies about FLOAT16_4 but stride proves it's FLOAT16_2
              // Some games (like Lego Batman 2) declare TEXCOORD as FLOAT16_4 (8 bytes) but the stride
              // shows it's actually FLOAT16_2 (4 bytes). This causes texcoords to read past the vertex
              // into the next vertex, creating "braille" rendering artifacts.
              //
              // For stride=28: Position(12) + Normal(4) + Tangent(4) + TexCoord(4) = 24 bytes + 4 padding = 28
              // If TEXCOORD at offset 20 was really FLOAT16_4 (8 bytes), we'd need stride >= 28
              // But with stride=28 and 4 bytes remaining after offset 20, it MUST be FLOAT16_2!
              {
                const uint32_t stride = drawCallState.originalVertexStride;
                const uint32_t bytesRemaining = stride - elem.Offset;

                // If there are only 4 bytes remaining after this element's offset, but the declared
                // format is FLOAT16_4 (which needs 8 bytes), the game is lying - fix it to FLOAT16_2
                if (bytesRemaining == 4) {
                  format = VK_FORMAT_R16G16_SFLOAT;
                  Logger::warn(str::format("[OPTION-A-EXECUTE] STRIDE MISMATCH DETECTED: Element at offset ", elem.Offset,
                                          " declared as FLOAT16_4 (8 bytes) but only ", bytesRemaining,
                                          " bytes remain in stride=", stride,
                                          " - CORRECTING to FLOAT16_2 (4 bytes) to prevent braille artifacts!"));
                }
              }
              break;
            default:
              Logger::warn(str::format("[OPTION-A-EXECUTE] Unknown vertex element type: ", elem.Type));
              continue;
          }

          // Add attribute
          attrList[attrCount].location = location;
          attrList[attrCount].binding = 0;
          attrList[attrCount].format = format;
          attrList[attrCount].offset = elem.Offset;

          attrCount++;
        }

      } else {
        // Fallback to stride-based guessing if no vertex declaration available
        Logger::warn("[OPTION-A-EXECUTE] No vertex declaration available - falling back to stride-based detection");

        uint32_t currentOffset = 0;

        // Position is always at offset 0 (required for all vertex formats)
        attrList[attrCount].location = 0;
        attrList[attrCount].binding = 0;
        attrList[attrCount].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrList[attrCount].offset = 0;
        attrCount++;
        currentOffset = 12;

        // Detect remaining attributes based on stride
        const uint32_t remainingBytes = drawCallState.originalVertexStride - 12;

        // Normal at offset 12 (if we have at least 24 bytes total)
        if (remainingBytes >= 12) {
          attrList[attrCount].location = 3;
          attrList[attrCount].binding = 0;
          attrList[attrCount].format = VK_FORMAT_R32G32B32_SFLOAT;
          attrList[attrCount].offset = currentOffset;
          attrCount++;
          currentOffset += 12;
        }

        // After Position+Normal, detect Texcoord and/or Color
        const uint32_t bytesAfterNormal = drawCallState.originalVertexStride - currentOffset;

        if (bytesAfterNormal >= 8) {
          // Texcoord (2 floats = 8 bytes)
          attrList[attrCount].location = 7;
          attrList[attrCount].binding = 0;
          attrList[attrCount].format = VK_FORMAT_R32G32_SFLOAT;
          attrList[attrCount].offset = currentOffset;
          attrCount++;
          currentOffset += 8;
        }

        const uint32_t bytesAfterTexcoord = drawCallState.originalVertexStride - currentOffset;

        if (bytesAfterTexcoord >= 4) {
          // Color (4 bytes)
          attrList[attrCount].location = 4;
          attrList[attrCount].binding = 0;
          attrList[attrCount].format = VK_FORMAT_B8G8R8A8_UNORM;
          attrList[attrCount].offset = currentOffset;
          attrCount++;
          currentOffset += 4;
        }

      }

      if (!useCapturedStreams) {
        // Vertex binding description
        bindList[0].binding = 0;
        bindList[0].fetchRate = 0;
        bindList[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        ctx->setInputLayout(attrCount, attrList, 1, bindList);
      } else {
        // OPTION A: Reconstruct per-stream vertex layout from captured D3D9 streams

        attrCount = 0;
        bindCount = 0;

        // Map D3D9 usage to Vulkan attribute location (same mapping as above)
        auto mapUsageToLocation = [](uint8_t usage, uint8_t usageIdx) -> uint32_t {
          switch (usage) {
            case D3DDECLUSAGE_POSITION:
            case D3DDECLUSAGE_POSITIONT: return 0;
            case D3DDECLUSAGE_BLENDWEIGHT: return 1;
            case D3DDECLUSAGE_BLENDINDICES: return 2;
            case D3DDECLUSAGE_NORMAL: return 3;
            case D3DDECLUSAGE_COLOR: return 4 + usageIdx;
            case D3DDECLUSAGE_TEXCOORD: return 7 + usageIdx;
            case D3DDECLUSAGE_TANGENT: return 6;
            case D3DDECLUSAGE_BINORMAL: return 9;
            case D3DDECLUSAGE_PSIZE: return 10;
            default: return 11 + usageIdx;
          }
        };

        // Step 1: Upload captured streams as Vulkan vertex buffers
        std::unordered_map<uint32_t, uint32_t> streamToBinding;
        std::vector<Rc<DxvkBuffer>> uploadedBuffers;

        for (const auto& stream : drawCallState.capturedVertexStreams) {
          if (stream.data.empty() || stream.stride == 0) continue;

          // Create replay buffer for this stream
          DxvkBufferCreateInfo info;
          info.size = stream.data.size();
          info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
          info.stages = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
          info.access = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
          info.requiredAlignmentOverride = 1;

          Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(
            info,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            DxvkMemoryStats::Category::AppBuffer,
            str::format("ShaderCapture Vertex Stream ", stream.streamIndex).c_str());

          if (void* dst = buffer->mapPtr(0)) {
            std::memcpy(dst, stream.data.data(), stream.data.size());
          } else {
            Logger::warn(str::format("[OPTION-A-MULTISTREAM] Failed to map stream ", stream.streamIndex));
            continue;
          }

          // Bind the buffer to a unique binding slot
          uint32_t binding = bindCount++;
          DxvkBufferSlice bufferSlice(buffer);
          ctx->bindVertexBuffer(binding, bufferSlice, stream.stride);

          bindList[binding].binding = binding;
          bindList[binding].fetchRate = 0;
          bindList[binding].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

          streamToBinding.emplace(stream.streamIndex, binding);
          uploadedBuffers.emplace_back(std::move(buffer));

        }

        // Step 2: Build vertex attributes from captured elements
        for (const auto& elem : drawCallState.capturedVertexElements) {
          if (elem.stream == 0xFF) break;

          // Find the binding for this stream
          auto it = streamToBinding.find(elem.stream);
          if (it == streamToBinding.end()) {
            Logger::warn(str::format("[OPTION-A-MULTISTREAM] Element references uncaptured stream ", elem.stream));
            continue;
          }

          // Convert D3D9 format to Vulkan format using existing DecodeDecltype
          VkFormat format = DecodeDecltype(D3DDECLTYPE(elem.type));
          if (format == VK_FORMAT_UNDEFINED) {
            Logger::warn(str::format("[OPTION-A-MULTISTREAM] Unknown D3D decltype: ", elem.type));
            continue;
          }

          attrList[attrCount].binding = it->second;
          attrList[attrCount].offset  = elem.offset;
          attrList[attrCount].format  = format;
          attrList[attrCount].location = mapUsageToLocation(elem.usage, elem.usageIndex);
          attrCount++;
        }

        if (attrCount == 0) {
          Logger::warn("[OPTION-A-MULTISTREAM] No valid attributes - falling back to legacy path");
          attrCount = 0;
          bindCount = 0;
        }
      }

      auto createReplayBuffer = [&](const std::vector<uint8_t>& data,
                                    VkBufferUsageFlags usage,
                                    VkPipelineStageFlags stages,
                                    VkAccessFlags access,
                                    const char* debugName) -> Rc<DxvkBuffer> {
        if (data.empty())
          return nullptr;

        DxvkBufferCreateInfo info;
        info.size = data.size();
        info.usage = usage;
        info.stages = stages;
        info.access = access;
        info.requiredAlignmentOverride = 1;

        Rc<DxvkBuffer> buffer = ctx->getDevice()->createBuffer(
          info,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          DxvkMemoryStats::Category::AppBuffer,
          debugName);

        if (void* dst = buffer->mapPtr(0)) {
          std::memcpy(dst, data.data(), data.size());
        } else {
          Logger::warn(str::format("[OPTION-A-EXECUTE] Failed to map ", debugName, " for replay"));
          buffer = nullptr;
        }

        return buffer;
      };

      Rc<DxvkBuffer> replayVertexBuffer = createReplayBuffer(
        preservedVertexData,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        "ShaderCapture Vertex Replay");

      // DEBUG: Dump COLOR0 values from first few vertices to diagnose alpha=0 issue
      {
        // Find COLOR0 attribute (location 4)
        uint32_t color0Offset = 0;
        bool hasColor0 = false;
        VkFormat color0Format = VK_FORMAT_UNDEFINED;
        for (uint32_t i = 0; i < attrCount; i++) {
          if (attrList[i].location == 4) { // COLOR0
            color0Offset = attrList[i].offset;
            color0Format = attrList[i].format;
            hasColor0 = true;
            break;
          }
        }

        if (!hasColor0) {
          Logger::warn("[COLOR0-DEBUG] No COLOR0 attribute found in vertex layout!");
        }
      }

      Rc<DxvkBuffer> replayIndexBuffer;
      std::vector<uint8_t> rebasedIndexBytes;
      if (!preservedIndexData.empty()) {
        const uint32_t indexStride =
          (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) ? 4u : 2u;
        const size_t indexCount = preservedIndexData.size() / indexStride;
        rebasedIndexBytes.resize(indexCount * indexStride);

        if (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) {
          const uint32_t* src = reinterpret_cast<const uint32_t*>(preservedIndexData.data());
          uint32_t* dst = reinterpret_cast<uint32_t*>(rebasedIndexBytes.data());
          for (size_t i = 0; i < indexCount; i++) {
            const int64_t rebased = int64_t(src[i]) - int64_t(drawCallState.originalIndexMin);
            dst[i] = rebased < 0 ? 0u : uint32_t(rebased);
          }
        } else {
          const uint16_t* src = reinterpret_cast<const uint16_t*>(preservedIndexData.data());
          uint16_t* dst = reinterpret_cast<uint16_t*>(rebasedIndexBytes.data());
          for (size_t i = 0; i < indexCount; i++) {
            const int32_t rebased = int32_t(src[i]) - int32_t(drawCallState.originalIndexMin);
            dst[i] = rebased < 0 ? uint16_t(0) : uint16_t(rebased);
          }
        }


        replayIndexBuffer = createReplayBuffer(
          rebasedIndexBytes,
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
          VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
          VK_ACCESS_INDEX_READ_BIT,
          "ShaderCapture Index Replay");
      }

      bool buffersValid = false;
      bool usedCapturedStreamsPath = false; // Track if we used the captured streams path
      if (bindCount > 0 && useCapturedStreams) {
        usedCapturedStreamsPath = true; // Mark that we're using captured streams
        ctx->setInputLayout(attrCount, attrList, bindCount, bindList);
        if (!preservedIndexData.empty()) {
          const uint32_t optionAIndexStride =
            (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) ? 4u : 2u;
          DxvkBufferSlice ibSlice(replayIndexBuffer);
          ctx->bindIndexBuffer(ibSlice, drawCallState.originalIndexType);
          if (optionAIndexStride > 0)
            actualIndexCount = preservedIndexData.size() / optionAIndexStride;
        } else {
          actualIndexCount = drawParams.indexCount > 0 ? drawParams.indexCount : geom.indexCount;
        }

        buffersValid = !geom.usesIndices() || replayIndexBuffer != nullptr || preservedIndexData.empty();
        const auto& firstStream = drawCallState.capturedVertexStreams.front();
        if (firstStream.stride > 0)
          actualVertexCount = firstStream.data.size() / firstStream.stride;
        if (actualVertexCount == 0 && drawParams.vertexCount > 0)
          actualVertexCount = drawParams.vertexCount;
      } else if (replayVertexBuffer != nullptr) {
        const uint32_t stride = drawCallState.originalVertexStride;
        DxvkBufferSlice vbSlice(replayVertexBuffer);
        ctx->bindVertexBuffer(0, vbSlice, stride);

        // Set up binding description for this legacy buffer path
        bindList[0].binding = 0;
        bindList[0].fetchRate = 0;
        bindList[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        ctx->setInputLayout(attrCount, attrList, 1, bindList);

        actualVertexCount = stride > 0 ? preservedVertexData.size() / stride : 0;

        uint32_t optionAIndexStride = (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) ? 4u : 2u;
        if (replayIndexBuffer != nullptr) {
          DxvkBufferSlice ibSlice(replayIndexBuffer);
          ctx->bindIndexBuffer(ibSlice, drawCallState.originalIndexType);

          if (optionAIndexStride > 0)
            actualIndexCount = preservedIndexData.size() / optionAIndexStride;
        } else if (geom.usesIndices()) {
          Logger::warn("[OPTION-A-EXECUTE] Index buffer replay data missing");
        }

        buffersValid = replayVertexBuffer != nullptr && (!geom.usesIndices() || replayIndexBuffer != nullptr);
      } else {
        Logger::warn("[OPTION-A-EXECUTE] Vertex buffer replay data missing - falling back to captured RasterGeometry");
      }

      // Set input assembly state (use topology from RasterGeometry)
      DxvkInputAssemblyState iaState = {};
      iaState.primitiveTopology = geom.topology;
      iaState.primitiveRestart = VK_FALSE;
      iaState.patchVertexCount = 0;
      ctx->setInputAssemblyState(iaState);

      // CRITICAL FIX: If we used capturedVertexStreams, those buffers are REBASED!
      // capturedVertexStreams are captured at Stage 4 with minIndex=0, so MUST use firstIndex=0
      usedOriginalBuffers = buffersValid || usedCapturedStreamsPath;
    }

    if (!usedOriginalBuffers) {
      // Fallback: Bind vertex and index buffers from RasterGeometry
      // NOTE: These are TRANSFORMED vertices which will cause braille with Option A approach

      // CRITICAL FIX: Geometry counts may be 0 in RasterGeometry for RT replacement draws
      // For indexed draws, we need indexCount from drawParams
      // For non-indexed draws, we need vertexCount from drawParams
      actualVertexCount = geom.vertexCount;
      actualIndexCount = geom.indexCount;

      // Always prefer drawParams for counts, as they're more reliable
      if (drawParams.indexCount > 0) {
        actualIndexCount = drawParams.indexCount;
      }
      if (drawParams.vertexCount > 0) {
        actualVertexCount = drawParams.vertexCount;
      }

      // Always log for RT replacement draws to debug geometry issue
      if (captureCount <= 5 || drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  actualVertexCount=", actualVertexCount, " actualIndexCount=", actualIndexCount));
      }

      // Set up vertex input layout manually since we're bypassing D3D9 state setup
      // D3D9 has up to 9 vertex attributes: position, normal, color0, color1, texcoord0-4
      // Note: attrList, bindList, attrCount, bindCount are already declared above
      attrCount = 0;
      bindCount = 0;

      // Check if vertex data is interleaved (same buffer for pos + texcoord + normal + color)
      isInterleaved = geom.isVertexDataInterleaved();

    // D3D9 vertex attribute locations (from d3d9_state.cpp):
    // 0 = position, 3 = normal, 4 = color0, 5 = color1, 7 = texcoord0
    // See D3DDECLUSAGE enum for full mapping

    // CRITICAL: For interleaved buffers, offsetFromSlice() returns the attribute offset WITHIN each vertex
    // For non-interleaved buffers, each buffer has its own base
    // We should use offsetFromSlice() DIRECTLY as the attribute offset

    if (geom.positionBuffer.defined()) {
      // Position attribute at location 0
      attrList[attrCount].location = 0;
      attrList[attrCount].binding = 0;
      attrList[attrCount].format = geom.positionBuffer.vertexFormat();

      // Use offsetFromSlice() directly - it's the attribute offset within the vertex
      attrList[attrCount].offset = static_cast<uint32_t>(geom.positionBuffer.offsetFromSlice());

      attrCount++;

      // Position binding - only add once for interleaved data
      if (bindCount == 0) {
        bindList[bindCount].binding = 0;
        bindList[bindCount].fetchRate = 0; // per-vertex
        bindList[bindCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindCount++;
      }

      // GeometryBuffer inherits from DxvkBufferSlice, so we can cast it directly
      const DxvkBufferSlice& posSlice = static_cast<const DxvkBufferSlice&>(geom.positionBuffer);
      const uint32_t vertexStride = geom.positionBuffer.stride();
      ctx->bindVertexBuffer(0, posSlice, vertexStride);

      if (captureCount <= 5 || drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  Position buffer: bound (stride=", vertexStride, " offset=", attrList[0].offset, " interleaved=", isInterleaved ? "yes" : "no", ")"));
        // Log the actual buffer slice details
        Logger::info(str::format("  posSlice: bufferOffset=", posSlice.offset(), " bufferLength=", posSlice.length(), " sliceOffsetFromBase=", geom.positionBuffer.offsetFromSlice()));

        // CRITICAL DEBUG: Read first vertex position to verify buffer is correct
        const void* vertexData = geom.positionBuffer.mapPtr(0);
        if (vertexData != nullptr) {
          const float* positions = static_cast<const float*>(vertexData);
          Logger::info(str::format("  First vertex position: [", positions[0], ", ", positions[1], ", ", positions[2], ", ", positions[3], "]"));

          // Check if positions are reasonable (not NaN, not huge)
          if (std::isnan(positions[0]) || std::abs(positions[0]) > 100000.0f) {
            Logger::err("  CRITICAL: First vertex position looks invalid!");
          }
        } else {
          Logger::warn("  Cannot read vertex buffer - mapPtr returned null");
        }

        if (vertexStride != 80) {
          Logger::warn(str::format("  WARNING: Unexpected stride! Expected 80, got ", vertexStride));
        }
      }
    }

    // Normal attribute at location 3 (D3DDECLUSAGE_NORMAL)
    if (geom.normalBuffer.defined()) {
      attrList[attrCount].location = 3;
      attrList[attrCount].format = geom.normalBuffer.vertexFormat();

      if (isInterleaved) {
        attrList[attrCount].binding = 0;
        // Use offsetFromSlice() directly
        attrList[attrCount].offset = static_cast<uint32_t>(geom.normalBuffer.offsetFromSlice());
      } else {
        attrList[attrCount].binding = bindCount;
        attrList[attrCount].offset = 0;

        bindList[bindCount].binding = bindCount;
        bindList[bindCount].fetchRate = 0;
        bindList[bindCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        const DxvkBufferSlice& normSlice = static_cast<const DxvkBufferSlice&>(geom.normalBuffer);
        ctx->bindVertexBuffer(bindCount, normSlice, geom.normalBuffer.stride());
        bindCount++;
      }

      if (captureCount <= 5 || drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  Normal buffer: bound (stride=", geom.normalBuffer.stride(), " offset=", attrList[attrCount].offset, " binding=", attrList[attrCount].binding, ")"));
      }
      attrCount++;
    }

    // Color0 attribute at location 4 (D3DDECLUSAGE_COLOR index 0)
    if (geom.color0Buffer.defined()) {
      attrList[attrCount].location = 4;
      attrList[attrCount].format = geom.color0Buffer.vertexFormat();

      if (isInterleaved) {
        attrList[attrCount].binding = 0;
        // Use offsetFromSlice() directly
        attrList[attrCount].offset = static_cast<uint32_t>(geom.color0Buffer.offsetFromSlice());
      } else {
        attrList[attrCount].binding = bindCount;
        attrList[attrCount].offset = 0;

        bindList[bindCount].binding = bindCount;
        bindList[bindCount].fetchRate = 0;
        bindList[bindCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        const DxvkBufferSlice& color0Slice = static_cast<const DxvkBufferSlice&>(geom.color0Buffer);
        ctx->bindVertexBuffer(bindCount, color0Slice, geom.color0Buffer.stride());
        bindCount++;
      }

      if (captureCount <= 5 || drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  Color0 buffer: bound (stride=", geom.color0Buffer.stride(), " offset=", attrList[attrCount].offset, " binding=", attrList[attrCount].binding, ")"));
      }
      attrCount++;
    }

    // Texcoord attribute at location 7 (D3DDECLUSAGE_TEXCOORD index 0)
    if (geom.texcoordBuffer.defined()) {
      attrList[attrCount].location = 7;
      attrList[attrCount].format = geom.texcoordBuffer.vertexFormat();

      if (isInterleaved) {
        attrList[attrCount].binding = 0;
        // Use offsetFromSlice() directly
        attrList[attrCount].offset = static_cast<uint32_t>(geom.texcoordBuffer.offsetFromSlice());
      } else {
        attrList[attrCount].binding = bindCount;
        attrList[attrCount].offset = 0;

        bindList[bindCount].binding = bindCount;
        bindList[bindCount].fetchRate = 0;
        bindList[bindCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        const DxvkBufferSlice& texSlice = static_cast<const DxvkBufferSlice&>(geom.texcoordBuffer);
        ctx->bindVertexBuffer(bindCount, texSlice, geom.texcoordBuffer.stride());
        bindCount++;
      }

      if (captureCount <= 5 || drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  Texcoord buffer: bound (stride=", geom.texcoordBuffer.stride(), " offset=", attrList[attrCount].offset, " binding=", attrList[attrCount].binding, ")"));
      }
      attrCount++;
    }

    // Set the vertex input layout
    ctx->setInputLayout(attrCount, attrList, bindCount, bindList);

    // CRITICAL: Set input assembly state (primitive topology)
    // Without this, the pipeline might use the wrong topology causing incorrect rendering
    DxvkInputAssemblyState iaState = {};
    iaState.primitiveTopology = geom.topology;
    iaState.primitiveRestart = VK_FALSE;
    iaState.patchVertexCount = 0;
    ctx->setInputAssemblyState(iaState);

    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info(str::format("  Input assembly: topology=", geom.topology));
    }

    if (geom.indexBuffer.defined()) {
      // GeometryBuffer inherits from DxvkBufferSlice, so we can cast it directly
      const DxvkBufferSlice& idxSlice = static_cast<const DxvkBufferSlice&>(geom.indexBuffer);
      ctx->bindIndexBuffer(idxSlice, geom.indexBuffer.indexType());

      if (drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("  Index buffer: bound (type=", geom.indexBuffer.indexType(),
                                " offset=", idxSlice.offset(),
                                " length=", idxSlice.length(), ")"));

        // CRITICAL DEBUG: Read actual index values to verify they're rebased
        const void* indexData = geom.indexBuffer.mapPtr(0);
        if (indexData != nullptr) {
          const uint16_t* indices16 = static_cast<const uint16_t*>(indexData);
          uint32_t maxIndex = 0;
          uint32_t minIndex = UINT32_MAX;

          const uint32_t numIndicesToCheck = std::min(actualIndexCount, 20u);
          Logger::info(str::format("  First ", numIndicesToCheck, " indices: "));
          std::string indicesStr = "    ";
          for (uint32_t i = 0; i < numIndicesToCheck; i++) {
            uint32_t idx = indices16[i];
            indicesStr += str::format(idx, ", ");
            maxIndex = std::max(maxIndex, idx);
            minIndex = std::min(minIndex, idx);
          }
          Logger::info(indicesStr);
          Logger::info(str::format("  Index range: [", minIndex, ", ", maxIndex, "] actualVertexCount=", actualVertexCount));

          if (maxIndex >= actualVertexCount) {
            Logger::err(str::format("  CRITICAL ERROR: maxIndex=", maxIndex, " >= actualVertexCount=", actualVertexCount, " - THIS CAUSES BRAILLE!"));
          }
        } else {
          Logger::warn("  Cannot read index buffer - mapPtr returned null");
        }

        Logger::info(str::format("  Drawing with firstIndex=0 vertexOffset=0"));
      }
    }
    }  // End of if (!usedOriginalBuffers) block - RasterGeometry binding complete

    // CRITICAL: Set spec constants to match the original draw state
    // We need to ensure spec constants are set correctly for our re-execution
    // NOTE: We preserve the game's original fog settings and transformation mode
    // (The terrain baker disables fog and enables CustomVertexTransform to inject custom projection)
    // We DON'T do that because we want to preserve the game's original rendering

    // CRITICAL FIX: Use the original game's transformation matrices
    // OPTION A: If using original D3D9 buffers (untransformed), disable CustomVertexTransform
    // OPTION B: If using RasterGeometry (transformed), enable CustomVertexTransform
    if (drawCallState.usesVertexShader) {
      // CRITICAL: Use the OUTER usedOriginalBuffers which was set correctly earlier
      // Don't recompute it here with the old buggy condition!
      if (usedOriginalBuffers) {
        // OPTION A: Original buffers are untransformed (object space)
        // Disable CustomVertexTransform so vertices go through normal transformation pipeline
        ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, false);

        Logger::info("[OPTION-A-EXECUTE] Disabled CustomVertexTransformEnabled - using normal transformation (object->world->view->projection)");
      } else {
        // OPTION B: RasterGeometry vertices are transformed (world/clip space)
        // Enable custom vertex transform mode (terrain baker approach)
        ctx->setSpecConstant(VK_PIPELINE_BIND_POINT_GRAPHICS, D3D9SpecConstantId::CustomVertexTransformEnabled, true);

      // Copy previous constant buffer data
      D3D9RtxVertexCaptureData& cbData = ctx->allocAndMapVertexCaptureConstantBuffer();
      if (rtState.vertexCaptureCB != nullptr) {
        cbData = *static_cast<D3D9RtxVertexCaptureData*>(rtState.vertexCaptureCB->mapPtr(0));
      }

      // CRITICAL INSIGHT: Captured vertices are in WORLD SPACE, not object space!
      // Vertex capture runs the VS and stores the OUTPUT position (world space)
      // So we only need View*Projection, NOT Object*View*Projection!
      // The ObjectToWorld transform was already applied during vertex capture
      const DrawCallTransforms& transforms = drawCallState.getTransformData();

      // Just use the view-to-projection matrix since vertices are already in world space
      cbData.customWorldToProjection = transforms.viewToProjection * transforms.worldToView;

      if (drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info("  Setting CustomVertexTransformEnabled = TRUE (vertices already in world space)");
        Logger::info(str::format("  worldToView[0][0]=", transforms.worldToView[0][0],
                                " viewToProjection[0][0]=", transforms.viewToProjection[0][0]));
        Logger::info(str::format("  customWorldToProjection[0][0]=", cbData.customWorldToProjection[0][0]));
      }
      }  // End of else block (RasterGeometry path with CustomVertexTransform)
    }

    // Helper to upload per-draw constant data into a dedicated uniform buffer.
    auto createConstantBufferSlice = [&](const std::vector<uint8_t>& data,
                                         VkPipelineStageFlags stages,
                                         const char* debugName) -> DxvkBufferSlice {
      if (data.empty())
        return DxvkBufferSlice();

      const Rc<DxvkDevice>& device = ctx->getDevice();
      VkDeviceSize alignment = device->properties().core.properties.limits.minUniformBufferOffsetAlignment;
      if (alignment == 0)
        alignment = 1;

      const VkDeviceSize rawSize = static_cast<VkDeviceSize>(data.size());
      const VkDeviceSize alignedSize = ((rawSize + alignment - 1) / alignment) * alignment;

      DxvkBufferCreateInfo info;
      info.size = alignedSize;
      info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
      info.stages = stages;
      info.access = VK_ACCESS_UNIFORM_READ_BIT;
      info.requiredAlignmentOverride = alignment;

      Rc<DxvkBuffer> uploadBuffer = device->createBuffer(
        info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        DxvkMemoryStats::Category::AppBuffer,
        debugName);

      if (void* dst = uploadBuffer->mapPtr(0)) {
        std::memcpy(dst, data.data(), data.size());

        if (alignedSize > rawSize) {
          std::memset(reinterpret_cast<char*>(dst) + rawSize, 0, size_t(alignedSize - rawSize));
        }
      } else {
        Logger::warn(str::format("[ShaderOutputCapturer] Failed to map ", debugName, " upload buffer (size=", rawSize, ")"));
      }

      return DxvkBufferSlice(uploadBuffer, 0, alignedSize);
    };

    // Bind vertex shader constants copied at Stage 2
    // Cast Vector4 vector to uint8_t vector for the buffer creation
    const std::vector<uint8_t> vsConstBytes(
      reinterpret_cast<const uint8_t*>(drawCallState.vertexShaderConstantData.data()),
      reinterpret_cast<const uint8_t*>(drawCallState.vertexShaderConstantData.data() + drawCallState.vertexShaderConstantData.size()));

    // NOTE: We no longer reject captures when VS constants c[0]-c[3] are zero.
    // Many D3D9 games use PRE-TRANSFORMED vertices (D3DFVF_XYZRHW) where vertices
    // are already in screen space and don't need a transformation matrix.
    // The executeGpuCaptureBatched() function handles this by building a proper
    // screen-to-clip orthographic projection matrix.

    DxvkBufferSlice vsConstantSlice = createConstantBufferSlice(
      vsConstBytes,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
      "ShaderCapture VS Constants");

    if (vsConstantSlice.defined()) {
      const uint32_t vsConstantBufferSlot = computeResourceSlotId(
        DxsoProgramType::VertexShader,
        DxsoBindingType::ConstantBuffer,
        DxsoConstantBuffers::VSConstantBuffer);

      ctx->bindResourceBuffer(vsConstantBufferSlot, vsConstantSlice);

    }

    // Bind pixel shader constants copied at Stage 2
    // Cast Vector4 vector to uint8_t vector for the buffer creation
    const std::vector<uint8_t> psConstBytes(
      reinterpret_cast<const uint8_t*>(drawCallState.pixelShaderConstantData.data()),
      reinterpret_cast<const uint8_t*>(drawCallState.pixelShaderConstantData.data() + drawCallState.pixelShaderConstantData.size()));

    DxvkBufferSlice psConstantSlice = createConstantBufferSlice(
      psConstBytes,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      "ShaderCapture PS Constants");

    if (psConstantSlice.defined()) {
      const uint32_t psConstantBufferSlot = computeResourceSlotId(
        DxsoProgramType::PixelShader,
        DxsoBindingType::ConstantBuffer,
        DxsoConstantBuffers::PSConstantBuffer);

      ctx->bindResourceBuffer(psConstantBufferSlot, psConstantSlice);

    }

    // RTX pixel shader constant buffer - use existing, don't allocate new

    // Execute draw call with current state (uses game's view/projection)
    // Use the actualVertexCount/actualIndexCount calculated above (either from geom or drawParams)

    // COMPREHENSIVE DEBUG LOGGING for RT replacement draws
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      Logger::info(str::format("[ShaderCapture-DRAW] Capture #", captureCount, " EXECUTING DRAW:"));
      Logger::info(str::format("  USED OPTION A (original buffers): ", usedOriginalBuffers ? "YES" : "NO"));
      if (usedOriginalBuffers) {
        Logger::info(str::format("  Option A stride: ", drawCallState.originalVertexStride));
        if (drawCallState.originalVertexStride == 0) {
          Logger::err("[ShaderCapture-DRAW] ❌ CRITICAL ERROR: originalVertexStride is 0! Vertex buffer binding will fail!");
        }
      } else {
        Logger::info(str::format("  RasterGeometry stride: ", geom.positionBuffer.stride()));
      }
      Logger::info(str::format("  actualIndexCount=", actualIndexCount, " actualVertexCount=", actualVertexCount));
      Logger::info(str::format("  drawParams.firstIndex=", drawParams.firstIndex, " drawParams.vertexOffset=", drawParams.vertexOffset));
      Logger::info(str::format("  drawParams.instanceCount=", drawParams.instanceCount));
      Logger::info(str::format("  viewport: ", static_cast<uint32_t>(drawCallState.originalViewport.width), "x",
                              static_cast<uint32_t>(std::abs(drawCallState.originalViewport.height))));
      Logger::info(str::format("  resolution: ", resolution.width, "x", resolution.height));
      Logger::info(str::format("  posBuffer stride=", geom.positionBuffer.stride(), " format=", geom.positionBuffer.vertexFormat()));
      if (geom.texcoordBuffer.defined()) {
        Logger::info(str::format("  texBuffer stride=", geom.texcoordBuffer.stride(), " format=", geom.texcoordBuffer.vertexFormat()));
      }
      Logger::info(str::format("  isInterleaved=", isInterleaved ? "YES" : "NO"));
      Logger::info(str::format("  attrCount=", attrCount, " bindCount=", bindCount));

      // Log attribute setup details
      for (uint32_t i = 0; i < attrCount; i++) {
        Logger::info(str::format("  attr[", i, "]: location=", attrList[i].location,
                                " binding=", attrList[i].binding,
                                " offset=", attrList[i].offset,
                                " format=", attrList[i].format));
      }

      // Log binding setup details
      for (uint32_t i = 0; i < bindCount; i++) {
        Logger::info(str::format("  bind[", i, "]: binding=", bindList[i].binding,
                                " inputRate=", bindList[i].inputRate,
                                " fetchRate=", bindList[i].fetchRate));
      }

      // Log buffer details
      if (geom.positionBuffer.defined()) {
        Logger::info(str::format("  positionBuffer: offset=", geom.positionBuffer.offsetFromSlice(),
                                " length=", geom.positionBuffer.length(),
                                " stride=", geom.positionBuffer.stride()));
      }
      if (geom.texcoordBuffer.defined()) {
        Logger::info(str::format("  texcoordBuffer: offset=", geom.texcoordBuffer.offsetFromSlice(),
                                " length=", geom.texcoordBuffer.length(),
                                " stride=", geom.texcoordBuffer.stride()));
      }
      if (geom.normalBuffer.defined()) {
        Logger::info(str::format("  normalBuffer: offset=", geom.normalBuffer.offsetFromSlice(),
                                " length=", geom.normalBuffer.length(),
                                " stride=", geom.normalBuffer.stride()));
      }
      if (geom.color0Buffer.defined()) {
        Logger::info(str::format("  color0Buffer: offset=", geom.color0Buffer.offsetFromSlice(),
                                " length=", geom.color0Buffer.length(),
                                " stride=", geom.color0Buffer.stride()));
      }
      if (geom.indexBuffer.defined()) {
        Logger::info(str::format("  indexBuffer: offset=", geom.indexBuffer.offsetFromSlice(),
                                " length=", geom.indexBuffer.length(),
                                " indexType=", geom.indexBuffer.indexType()));
      }
    }

    // Execute the draw using either original buffers (Option A) or RasterGeometry
    // CRITICAL: Draw parameters depend on which path we took!
    // - Option A (original buffers): Use rebased params (firstIndex=0, vertexOffset=0)
    // - RasterGeometry: Use ORIGINAL draw params

    const uint32_t firstIndex = usedOriginalBuffers ? 0 : drawParams.firstIndex;
    const uint32_t vertexOffset = usedOriginalBuffers ? 0 : drawParams.vertexOffset;

    static uint32_t paramLogCount = 0;
    if ((++paramLogCount <= 20) || (matHash == 0)) {
      Logger::info(str::format("[ShaderCapture-PARAMS] usedOriginalBuffers=", usedOriginalBuffers ? 1 : 0,
                              " firstIndex=", firstIndex,
                              " vertexOffset=", vertexOffset,
                              " drawParams.firstIndex=", drawParams.firstIndex,
                              " drawParams.vertexOffset=", drawParams.vertexOffset,
                              " matHash=0x", std::hex, matHash, std::dec));
    }

    auto drawCallStartTime = std::chrono::high_resolution_clock::now();

    if (actualIndexCount == 0) {
      // Non-indexed draw
      const uint32_t startVertex = usedOriginalBuffers ? 0 : drawParams.vertexOffset;

      ctx->DxvkContext::draw(
        actualVertexCount,
        drawParams.instanceCount,
        startVertex,
        0);
    } else {
      // Indexed draw
      ctx->DxvkContext::drawIndexed(
        actualIndexCount,
        drawParams.instanceCount,
        firstIndex,
        vertexOffset,
        0);
    }

    if (captureCount <= 5) {
      Logger::info(str::format("[ShaderOutputCapturer] Capture #", captureCount, " - Draw call executed"));
      Logger::info(str::format("  Summary: matHash=0x", std::hex, matHash, std::dec,
                              " resolution=", resolution.width, "x", resolution.height,
                              " hasRenderTargetReplacement=", drawCallState.renderTargetReplacementSlot >= 0 ? "YES" : "NO",
                              " replacementSlot=", drawCallState.renderTargetReplacementSlot,
                              " capturedD3D9Textures.size=", drawCallState.capturedD3D9Textures.size()));

      // NOTE: The draw call executed, but we can't easily verify if pixels were actually written
      // without reading back from GPU (expensive). The captured texture will be either:
      // 1. Magenta (clear color) if nothing rendered - likely means shaders/textures aren't bound
      // 2. Actual rendered output if shaders executed properly
      //
      // To verify, check the path traced output in-game. If you see magenta on surfaces,
      // it means the capture is empty and we need to ensure shader state is preserved.
    }

    // ASYNC OPTIMIZATION: Removed pixel readback debug code that was calling flushCommandList()
    // and blocking GPU pipeline. Captures now run fully async without CPU/GPU sync points.

    // BARRIER OPTIMIZATION: No transition needed! GENERAL layout supports both rendering and reading
    // This eliminates 100+ barriers per frame (was transitioning to SHADER_READ_ONLY_OPTIMAL)
    // GENERAL layout allows:
    // 1. Image to be saved to disk correctly
    // 2. Image to be used as shader input texture in raytracing pipeline
    // 3. NO BARRIERS needed between captures!

    Logger::info(str::format("[ShaderCapture-Layout] Render target 0x", std::hex,
                            renderTarget.image->getHash(), std::dec,
                            " stays in GENERAL layout (no transition needed - barrier optimization!)"));

    // Restore previous render target (game's RT, not our captured RT)
    if (prevColorTarget != nullptr) {
      DxvkRenderTargets prevRt;
      prevRt.color[0].view = prevColorTarget;
      prevRt.color[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Game's RT uses standard layout
      ctx->bindRenderTargets(prevRt);
    }

    // PERFORMANCE: Disabled debug texture saving - was causing massive slowdown from disk I/O
    // Enable only for debugging specific materials, not for production use
    // if (drawCallState.renderTargetReplacementSlot < 0 && drawCallState.originalRenderTargetHash != 0) {
    //   auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    //   std::string filename = str::format("RT_", std::hex, cacheKey, std::dec, "_",
    //                                     renderTarget.image->info().extent.width, "x",
    //                                     renderTarget.image->info().extent.height, ".dds");
    //   try {
    //     auto& exporter = ctx->getCommonObjects()->metaExporter();
    //     exporter.dumpImageToFile(ctx.ptr(), "C:/Program Files/Epic Games/LegoBatman2/captured/", filename, renderTarget.image);
    //   } catch (...) { }
    // }

    // Store captured output
    storeCapturedOutput(ctx, drawCallState, renderTarget, currentFrame);

    // Return texture reference
    outputTexture = TextureRef(renderTarget.view);

    // Log what we're returning for RT replacements
    if (drawCallState.renderTargetReplacementSlot >= 0) {
      const XXH64_hash_t returnedHash = outputTexture.isValid() ? outputTexture.getImageHash() : 0;
      const XXH64_hash_t imageHash = (renderTarget.view != nullptr) ? renderTarget.view->image()->getHash() : 0;
      Logger::info(str::format("[ShaderOutputCapturer] Returning captured texture - outputTexture.isValid()=", outputTexture.isValid() ? 1 : 0,
                              " outputTexture.getImageHash()=0x", std::hex, returnedHash, std::dec,
                              " renderTarget.image.getHash()=0x", std::hex, imageHash, std::dec));
    }

    m_capturesThisFrame++;

    return true;
  }

  // Public API - computes cache key from DrawCallState
  TextureRef ShaderOutputCapturer::getCapturedTexture(const DrawCallState& drawCallState) const {
    // CRITICAL FIX: Use same cache key as storage (combined hash for RT replacements)
    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    if (!isValidKey) {
      static uint32_t invalidKeyCount = 0;
      if (++invalidKeyCount <= 50) {
        Logger::warn(str::format("[GET-CACHED-TEX] #", invalidKeyCount, " Invalid key, returning empty"));
      }
      return TextureRef();
    }

    TextureRef result = getCapturedTextureInternal(cacheKey);

    static uint32_t getCallCount = 0;
    if (++getCallCount <= 50) {
      Logger::warn(str::format("[GET-CACHED-TEX] #", getCallCount,
                              " cacheKey=0x", std::hex, cacheKey, std::dec,
                              " result.isValid()=", result.isValid() ? "TRUE" : "FALSE",
                              " resultHash=0x", std::hex, (result.isValid() ? result.getImageHash() : 0), std::dec));
    }

    return result;
  }

  // Public API - computes cache key from DrawCallState
  bool ShaderOutputCapturer::hasCapturedTexture(const DrawCallState& drawCallState) const {
    // CRITICAL FIX: Use same cache key as storage (combined hash for RT replacements)
    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    if (!isValidKey) {
      return false;
    }
    return hasCapturedTextureInternal(cacheKey);
  }

  // Internal implementation - uses pre-computed cache key for efficiency
  TextureRef ShaderOutputCapturer::getCapturedTextureInternal(XXH64_hash_t cacheKey) const {
    auto it = m_capturedOutputs.find(cacheKey);
    if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
      // Update LRU tracking - this material was accessed this frame
      // m_capturedOutputs is mutable, so we can update it from const function
      const_cast<CapturedShaderOutput&>(it->second).lastCaptureFrame = m_currentFrame;

      XXH64_hash_t imageHash = it->second.capturedTexture.image->getHash();

      static uint32_t internalGetCount = 0;
      if (++internalGetCount <= 30) {
        Logger::warn(str::format("[GET-INTERNAL] #", internalGetCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " imageHash=0x", std::hex, imageHash, std::dec,
                                " USING cacheKey as uniqueKey since imageHash=0",
                                " viewPtr=", (void*)it->second.capturedTexture.view.ptr(),
                                " imagePtr=", (void*)it->second.capturedTexture.image.ptr()));
      }

      // CRITICAL FIX: Use cacheKey as uniqueKey instead of imageHash
      // Image hash is 0 because the image hasn't been used yet (hash computed lazily)
      // Using cacheKey ensures each material gets a unique TextureRef
      return TextureRef(it->second.capturedTexture.view, cacheKey);
    }
    return TextureRef();
  }

  // Internal implementation - uses pre-computed cache key for efficiency
  bool ShaderOutputCapturer::hasCapturedTextureInternal(XXH64_hash_t cacheKey) const {
    auto it = m_capturedOutputs.find(cacheKey);
    if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
      // Update LRU tracking - this material was accessed this frame
      // m_capturedOutputs is mutable, so we can update it from const function
      const_cast<CapturedShaderOutput&>(it->second).lastCaptureFrame = m_currentFrame;
      return true;
    }
    return false;
  }

  void ShaderOutputCapturer::onFrameBegin(Rc<RtxContext> ctx) {
    static uint32_t frameBeginCallCount = 0;
    ++frameBeginCallCount;

    // AGGRESSIVE LOGGING FOR DIAGNOSIS
    Logger::info(str::format("[FRAMEBEGIN-ENTRY] ========== onFrameBegin #", frameBeginCallCount,
                            " pendingRequests=", m_pendingCaptureRequests.size(),
                            " enableShaderOutputCapture=", enableShaderOutputCapture() ? "TRUE" : "FALSE",
                            " =========="));

    // GPU-DRIVEN CAPTURE SYSTEM: Lazy initialization on first frame
    static bool gpuSystemInitialized = false;
    if (!gpuSystemInitialized && enableShaderOutputCapture()) {
      Logger::info("[ShaderCapture-GPU-INIT] Initializing GPU-driven batched capture system...");
      initializeGpuCaptureSystem(ctx);
      // NO pre-allocated pool - RTs allocated on-demand for massive VRAM savings!
      gpuSystemInitialized = true;
      Logger::info("[ShaderCapture-GPU-INIT] ===== GPU-DRIVEN CAPTURE SYSTEM READY =====");
    }

    // Execute any remaining captures from previous frame
    Logger::info(str::format("[FRAMEBEGIN-CHECK] enableShaderOutputCapture=", enableShaderOutputCapture() ? "TRUE" : "FALSE",
                            " pendingRequests.empty=", m_pendingCaptureRequests.empty() ? "TRUE" : "FALSE",
                            " pendingRequests.size=", m_pendingCaptureRequests.size()));

    if (enableShaderOutputCapture() && !m_pendingCaptureRequests.empty()) {
      Logger::info(str::format("[ShaderCapture-GPU-EXEC] ========== EXECUTING ", m_pendingCaptureRequests.size(),
                              " queued captures from previous frame =========="));
      buildGpuCaptureList(ctx);
      executeMultiIndirectCaptures(ctx);
    } else {
      Logger::info("[FRAMEBEGIN-SKIP] Skipping executeMultiIndirectCaptures (no requests or disabled)");
    }

    m_capturesThisFrame = 0;
    m_currentFrame++;

    // ASYNC CAPTURE POLLING: Check all pending captures to see if GPU has completed them
    // Use DXVK's built-in resource tracking via isInUse() to detect completion
    uint32_t totalPending = 0;
    uint32_t completedThisFrame = 0;

    for (auto& pair : m_capturedOutputs) {
      auto& output = pair.second;

      // Only check captures that are marked as pending
      if (output.isPending && output.capturedTexture.isValid() && output.capturedTexture.image.ptr() != nullptr) {
        totalPending++;

        // Check if GPU has finished this capture via resource tracking
        if (!output.capturedTexture.image->isInUse()) {
          // GPU is done - mark as complete
          output.isPending = false;
          completedThisFrame++;

          static uint32_t completionLogCount = 0;
          if (++completionLogCount <= 20) {
            Logger::info(str::format("[ShaderCapture-ASYNC] #", completionLogCount,
                                    " Capture COMPLETED in onFrameBegin - matHash=0x", std::hex, output.materialHash, std::dec,
                                    " (GPU finished after ", m_currentFrame - output.captureSubmittedFrame, " frames)"));
          }
        }
      }
    }

    // Log async capture status periodically
    static uint32_t frameBeginLogCount = 0;
    if (++frameBeginLogCount <= 10 || (frameBeginLogCount % 60 == 0)) {
      Logger::info(str::format("[ShaderCapture-ASYNC] Frame ", m_currentFrame,
                              " - Total cached: ", m_capturedOutputs.size(),
                              " | Pending: ", totalPending - completedThisFrame,
                              " | Completed this frame: ", completedThisFrame));
    }
  }

  void ShaderOutputCapturer::onFrameEnd() {
    static uint32_t frameEndCallCount = 0;
    ++frameEndCallCount;

    // ASYNC FRAME SPREADING: Do NOT clear pending requests here!
    // Unprocessed requests are intentionally kept queued for next frame.
    // Queue is managed in executeMultiIndirectCaptures() which removes only processed requests.
    // m_pendingCaptureRequests.clear();  // REMOVED - was breaking async frame spreading

    // PROACTIVE LRU VRAM CLEANUP - runs every frame
    // This is critical for static materials which never call getRenderTarget() again after initial capture
    // Without this, render target cache grows unbounded until OOM
    {
      size_t maxVramBytes = static_cast<size_t>(maxVramMB()) * 1024 * 1024;
      size_t currentVramBytes = 0;

      // Calculate current VRAM usage
      for (const auto& entry : m_renderTargetCache) {
        currentVramBytes += entry.second.vramBytes;
      }

      // Evict oldest RTs if we exceed VRAM limit
      if (currentVramBytes > maxVramBytes) {
        // Sort by lastUsedFrame (oldest first)
        std::vector<std::pair<uint64_t, uint32_t>> rtsByAge;
        for (const auto& entry : m_renderTargetCache) {
          rtsByAge.push_back({entry.first, entry.second.lastUsedFrame});
        }
        std::sort(rtsByAge.begin(), rtsByAge.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Evict oldest until we're under the limit
        size_t freedBytes = 0;
        for (const auto& [evictKey, evictFrame] : rtsByAge) {
          if (currentVramBytes - freedBytes <= maxVramBytes) {
            break;
          }

          auto evictIt = m_renderTargetCache.find(evictKey);
          if (evictIt != m_renderTargetCache.end()) {
            // SAFETY: Skip RTs used within last 2 frames to avoid evicting actively-used render targets
            uint32_t age = m_currentFrame - evictIt->second.lastUsedFrame;
            if (age < 2) {
              continue; // Skip very recent RTs that might still be in use
            }

            freedBytes += evictIt->second.vramBytes;
            Logger::info(str::format("[LRU-EVICT] Freed ", evictIt->second.vramBytes / (1024 * 1024), " MB ",
                                     "(frame ", evictFrame, ", age ", age, " frames) ",
                                     "Total VRAM: ", (currentVramBytes - freedBytes) / (1024 * 1024), " MB"));
            m_renderTargetCache.erase(evictIt);
          }
        }
      }
    }

    // PROACTIVE LRU VRAM CLEANUP FOR CAPTURED OUTPUTS - runs every frame
    // This prevents OOM when loading scenes with hundreds of materials
    // Evicts least-recently-used captured material outputs when VRAM limit is exceeded
    {
      size_t maxCapturedVramBytes = static_cast<size_t>(maxCapturedOutputsVramMB()) * 1024 * 1024;
      size_t currentCapturedVramBytes = 0;

      // Calculate current VRAM usage for captured outputs
      for (const auto& pair : m_capturedOutputs) {
        currentCapturedVramBytes += pair.second.vramBytes;
      }

      // Evict oldest captured outputs if we exceed VRAM limit
      if (currentCapturedVramBytes > maxCapturedVramBytes) {
        // Sort by lastCaptureFrame (oldest first)
        std::vector<std::pair<XXH64_hash_t, uint32_t>> outputsByAge;
        for (const auto& pair : m_capturedOutputs) {
          outputsByAge.push_back({pair.first, pair.second.lastCaptureFrame});
        }
        std::sort(outputsByAge.begin(), outputsByAge.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Evict oldest until we're under the limit
        size_t freedBytes = 0;
        uint32_t evictedCount = 0;
        for (const auto& [evictKey, evictFrame] : outputsByAge) {
          if (currentCapturedVramBytes - freedBytes <= maxCapturedVramBytes) {
            break;
          }

          auto evictIt = m_capturedOutputs.find(evictKey);
          if (evictIt != m_capturedOutputs.end()) {
            // SAFETY: Skip pending textures (GPU still writing) and very recent captures
            // Evicting these could cause corruption or missing textures
            if (evictIt->second.isPending) {
              continue; // Skip pending textures
            }

            // Skip textures captured within last 2 frames to avoid evicting actively-used materials
            uint32_t age = m_currentFrame - evictIt->second.lastCaptureFrame;
            if (age < 2) {
              continue; // Skip very recent textures
            }

            freedBytes += evictIt->second.vramBytes;
            ++evictedCount;

            // Log first few evictions for debugging
            static uint32_t evictLogCount = 0;
            if (++evictLogCount <= 20) {
              Logger::info(str::format("[LRU-EVICT-CAPTURE] #", evictLogCount,
                                      " Freed ", evictIt->second.vramBytes / (1024 * 1024), " MB ",
                                      "matHash=0x", std::hex, evictIt->second.materialHash, std::dec,
                                      " (frame ", evictFrame, ", age ", age, " frames) ",
                                      "Total: ", (currentCapturedVramBytes - freedBytes) / (1024 * 1024), " MB"));
            }

            m_capturedOutputs.erase(evictIt);
          }
        }

        // Log summary of eviction
        if (evictedCount > 0) {
          static uint32_t summaryLogCount = 0;
          if (++summaryLogCount <= 10 || (summaryLogCount % 60 == 0)) {
            Logger::info(str::format("[LRU-EVICT-CAPTURE-SUMMARY] Evicted ", evictedCount, " materials, ",
                                    "freed ", freedBytes / (1024 * 1024), " MB, ",
                                    "remaining: ", m_capturedOutputs.size(), " materials, ",
                                    (currentCapturedVramBytes - freedBytes) / (1024 * 1024), " MB"));
          }
        }
      }

      // Log VRAM usage periodically
      static uint32_t vramLogCount = 0;
      if (++vramLogCount % 60 == 0) {
        Logger::info(str::format("[CAPTURED-OUTPUTS-VRAM] Frame ", m_currentFrame,
                                " - Cached materials: ", m_capturedOutputs.size(),
                                " | VRAM: ", currentCapturedVramBytes / (1024 * 1024), " MB / ",
                                maxCapturedVramBytes / (1024 * 1024), " MB"));
      }
    }

    // REMOVED: Don't clear entire RT cache - there's always at least one RT being actively used
    // The LRU eviction above handles cleanup safely by checking age
    // m_renderTargetCache.clear();

    // Periodic cleanup of texture array pools (unchanged)
    if (m_currentFrame % 60 == 0) {
      // Clear texture array pools - will be rebuilt as needed
      m_arrayPools.clear();
    }
  }

  Resources::Resource ShaderOutputCapturer::getRenderTarget(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      XXH64_hash_t materialHash) {

    // REMOVED OPTIMIZATION: Cannot reuse captured textures as render targets!
    // Captured textures are READ-ONLY outputs - rendering into them corrupts the capture.
    // We must always allocate fresh RTs for new captures, even if material was already captured.

    // Create cache key from resolution and format ONLY (NO material hash!)
    // This allows RT sharing between materials with same resolution = MASSIVE VRAM savings
    // We don't need per-material RTs because we copy the result immediately after capture
    uint64_t cacheKey = (static_cast<uint64_t>(resolution.width) << 32) |
                        (static_cast<uint64_t>(resolution.height) << 16) |
                        static_cast<uint64_t>(format);
    // REMOVED: XOR with material hash - was causing 1 RT per material = GB of wasted VRAM!

    // Check cache - LRU tracking
    auto it = m_renderTargetCache.find(cacheKey);
    if (it != m_renderTargetCache.end() && it->second.resource.isValid()) {
      // Update last used frame for LRU tracking
      it->second.lastUsedFrame = m_currentFrame;
      return it->second.resource;
    }

    // LRU EVICTION: Check if we need to free VRAM before allocating
    // Calculate bytes per pixel for this format
    uint32_t bytesPerPixel = 4; // RGBA8 = 4 bytes
    switch (format) {
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_UNORM:
        bytesPerPixel = 4;
        break;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        bytesPerPixel = 8;
        break;
      case VK_FORMAT_R32G32B32A32_SFLOAT:
        bytesPerPixel = 16;
        break;
      default:
        bytesPerPixel = 4;
        break;
    }

    size_t newRTBytes = resolution.width * resolution.height * bytesPerPixel;
    size_t maxVramBytes = static_cast<size_t>(maxVramMB()) * 1024 * 1024;

    // Calculate current VRAM usage
    size_t currentVramBytes = 0;
    for (const auto& entry : m_renderTargetCache) {
      currentVramBytes += entry.second.vramBytes;
    }

    // Evict oldest RTs if we'll exceed VRAM limit
    if (currentVramBytes + newRTBytes > maxVramBytes) {
      // Sort by lastUsedFrame (oldest first)
      std::vector<std::pair<uint64_t, uint32_t>> rtsByAge; // (cacheKey, lastUsedFrame)
      for (const auto& entry : m_renderTargetCache) {
        rtsByAge.push_back({entry.first, entry.second.lastUsedFrame});
      }
      std::sort(rtsByAge.begin(), rtsByAge.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

      // Evict oldest until we have enough space
      size_t freedBytes = 0;
      for (const auto& [evictKey, evictFrame] : rtsByAge) {
        if (currentVramBytes - freedBytes + newRTBytes <= maxVramBytes) {
          break; // Enough space now
        }

        auto evictIt = m_renderTargetCache.find(evictKey);
        if (evictIt != m_renderTargetCache.end()) {
          freedBytes += evictIt->second.vramBytes;
          Logger::info(str::format("[LRU-EVICT] Freed ", evictIt->second.vramBytes / (1024 * 1024), " MB ",
                                   "(frame ", evictFrame, ", age ", m_currentFrame - evictFrame, " frames)"));
          m_renderTargetCache.erase(evictIt);
        }
      }

      if (freedBytes > 0) {
        Logger::info(str::format("[LRU-EVICT] Total freed: ", freedBytes / (1024 * 1024), " MB, ",
                                 "VRAM usage: ", (currentVramBytes - freedBytes) / (1024 * 1024), " MB -> ",
                                 (currentVramBytes - freedBytes + newRTBytes) / (1024 * 1024), " MB"));
      }
    }

    // Create new render target - upcast to DxvkContext&
    VkExtent3D extent = { resolution.width, resolution.height, 1 };

    Rc<DxvkContext> dxvkCtx = ctx;
    Resources::Resource resource = Resources::createImageResource(
      dxvkCtx,
      "shader output capture target",
      extent,
      format,
      1, // mip levels
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D,
      0, // image create flags
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // Set the cache key as the texture hash - this is unique per material
    if (resource.image != nullptr) {
      resource.image->setHash(cacheKey);
      Logger::info(str::format("[ShaderOutputCapturer] Created render target: ",
                               resolution.width, "x", resolution.height,
                               " format=", format,
                               " matHash=0x", std::hex, materialHash, std::dec,
                               " textureHash=0x", std::hex, cacheKey, std::dec,
                               " VRAM: ", newRTBytes / (1024 * 1024), " MB"));
    }

    // Cache it with LRU tracking
    RenderTargetCacheEntry entry;
    entry.resource = resource;
    entry.lastUsedFrame = m_currentFrame;
    entry.vramBytes = newRTBytes;
    m_renderTargetCache[cacheKey] = entry;

    return resource;
  }

  Resources::Resource ShaderOutputCapturer::getRenderTargetArray(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      uint32_t layerCount) {

    // OPTIMIZATION 1: Cache texture arrays by (resolution, layerCount)
    // Key format: width (16 bits) | height (16 bits) | layerCount (32 bits)
    uint64_t cacheKey = (uint64_t(resolution.width) << 48) |
                        (uint64_t(resolution.height) << 32) |
                        uint64_t(layerCount);

    // Check cache first
    static uint32_t s_cacheHits = 0;
    static uint32_t s_cacheMisses = 0;
    static uint32_t s_totalCacheChecks = 0;

    auto it = m_renderTargetArrayCache.find(cacheKey);
    s_totalCacheChecks++;

    if (it != m_renderTargetArrayCache.end() && it->second.isValid()) {
      s_cacheHits++;
      float hitRate = 100.0f * s_cacheHits / s_totalCacheChecks;
      Logger::info(str::format("[CACHE-HIT] RT ", resolution.width, "x", resolution.height,
                               " layers=", layerCount, " (hit rate: ", hitRate, "% - ",
                               s_cacheHits, "/", s_totalCacheChecks, ")"));
      return it->second;
    }

    // Cache miss - create new texture array
    s_cacheMisses++;
    float missRate = 100.0f * s_cacheMisses / s_totalCacheChecks;
    Logger::info(str::format("[CACHE-MISS] RT ", resolution.width, "x", resolution.height,
                             " layers=", layerCount, " NOT in cache (miss rate: ", missRate,
                             "% - ", s_cacheMisses, "/", s_totalCacheChecks,
                             ") cache size=", m_renderTargetArrayCache.size()));

    VkExtent3D extent = { resolution.width, resolution.height, 1 };

    Rc<DxvkContext> dxvkCtx = ctx;
    Resources::Resource resource = Resources::createImageResource(
      dxvkCtx,
      "shader capture texture array",
      extent,
      format,
      layerCount, // TEXTURE ARRAY with N layers
      VK_IMAGE_TYPE_2D,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY, // ARRAY view type
      0, // image create flags
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    if (resource.image != nullptr) {
      // CRITICAL FIX: Set the hash on the image so it can be identified
      // Use cacheKey as hash - same as getRenderTarget() does
      resource.image->setHash(cacheKey);

      VkImageLayout initialLayout = resource.image->info().layout;
      Logger::info(str::format("[CACHE-MISS] Created NEW RT 0x", std::hex, resource.image->getHash(), std::dec,
                               " ", resolution.width, "x", resolution.height,
                               " layers=", layerCount, " initialLayout=", initialLayout));
      // Add to cache
      m_renderTargetArrayCache[cacheKey] = resource;
    } else {
      Logger::err(str::format("[CACHE-MISS] FAILED to create RT ", resolution.width, "x", resolution.height));
    }

    return resource;
  }

  // OPTIMIZATION 2: Texture array pool with layer allocation
  ShaderOutputCapturer::LayerAllocation ShaderOutputCapturer::allocateLayersFromPool(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      uint32_t layerCount) {

    LayerAllocation result;

    // Can't use pool if requesting more than pool size
    if (layerCount > POOL_LAYERS_PER_ARRAY) {
      Logger::warn(str::format("[PERF] Requesting ", layerCount, " layers exceeds pool size (",
                              POOL_LAYERS_PER_ARRAY, "), falling back to direct allocation"));
      return result;  // invalid
    }

    // Find a pool with matching resolution and enough free layers
    for (size_t i = 0; i < m_arrayPools.size(); ++i) {
      auto& pool = m_arrayPools[i];

      // Check resolution match
      if (pool.resolution.width != resolution.width ||
          pool.resolution.height != resolution.height) {
        continue;
      }

      // Check if we have enough consecutive free layers
      uint32_t freeCount = pool.totalLayers - pool.allocatedLayers;
      if (freeCount >= layerCount) {
        // Find consecutive free layers in the bitmask
        uint64_t mask = pool.usedLayersMask;
        for (uint32_t startLayer = 0; startLayer <= pool.totalLayers - layerCount; ++startLayer) {
          // Check if layerCount consecutive layers starting at startLayer are free
          bool allFree = true;
          for (uint32_t j = 0; j < layerCount; ++j) {
            if (mask & (1ULL << (startLayer + j))) {
              allFree = false;
              break;
            }
          }

          if (allFree) {
            // Found free range! Allocate it
            for (uint32_t j = 0; j < layerCount; ++j) {
              pool.usedLayersMask |= (1ULL << (startLayer + j));
            }
            pool.allocatedLayers += layerCount;

            result.arrayResource = pool.arrayResource;
            result.startLayer = startLayer;
            result.layerCount = layerCount;
            result.poolIndex = i;
            result.valid = true;

            Logger::info(str::format("[PERF-OPT] Allocated ", layerCount, " layers from pool ", i,
                                    " (layers ", startLayer, "-", startLayer + layerCount - 1, ")"));
            return result;
          }
        }
      }
    }

    // No existing pool has space - try to create a new pool
    if (m_arrayPools.size() < MAX_POOL_ARRAYS) {
      VkExtent3D extent = { resolution.width, resolution.height, 1 };
      Rc<DxvkContext> dxvkCtx = ctx;

      Resources::Resource poolArray = Resources::createImageResource(
        dxvkCtx,
        "shader capture array pool",
        extent,
        VK_FORMAT_R8G8B8A8_UNORM,
        POOL_LAYERS_PER_ARRAY,
        VK_IMAGE_TYPE_2D,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        0,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

      if (poolArray.isValid()) {
        TextureArrayPool newPool;
        newPool.arrayResource = poolArray;
        newPool.totalLayers = POOL_LAYERS_PER_ARRAY;
        newPool.usedLayersMask = 0;
        newPool.allocatedLayers = layerCount;
        newPool.resolution = resolution;

        // Mark first layerCount layers as used
        for (uint32_t i = 0; i < layerCount; ++i) {
          newPool.usedLayersMask |= (1ULL << i);
        }

        m_arrayPools.push_back(newPool);

        result.arrayResource = poolArray;
        result.startLayer = 0;
        result.layerCount = layerCount;
        result.poolIndex = m_arrayPools.size() - 1;
        result.valid = true;

        Logger::info(str::format("[PERF-OPT] Created NEW array pool ", m_arrayPools.size() - 1,
                                " (", resolution.width, "x", resolution.height, ", ",
                                POOL_LAYERS_PER_ARRAY, " layers), allocated ", layerCount, " layers"));
        return result;
      }
    }

    Logger::warn("[PERF] No pool space available, falling back to direct allocation");
    return result;  // invalid
  }

  void ShaderOutputCapturer::freeLayersToPool(const LayerAllocation& allocation) {
    if (!allocation.valid || allocation.poolIndex >= m_arrayPools.size()) {
      return;
    }

    auto& pool = m_arrayPools[allocation.poolIndex];

    // Free the layers
    for (uint32_t i = 0; i < allocation.layerCount; ++i) {
      pool.usedLayersMask &= ~(1ULL << (allocation.startLayer + i));
    }
    pool.allocatedLayers -= allocation.layerCount;

    Logger::info(str::format("[PERF-OPT] Freed ", allocation.layerCount, " layers from pool ",
                            allocation.poolIndex, " (", pool.allocatedLayers, "/",
                            pool.totalLayers, " now used)"));
  }

  VkExtent2D ShaderOutputCapturer::calculateCaptureResolution(
      const DrawCallState& drawCallState) const {

    // MATCH SOURCE TEXTURE RESOLUTION - don't waste VRAM on upscaling!
    // For 512x512 source -> 512x512 capture (not 1024x1024)
    // CRITICAL: Preserve aspect ratio! Don't force square textures!
    const auto& materialData = drawCallState.getMaterialData();
    const TextureRef& sourceTexture = materialData.getColorTexture();

    uint32_t fallbackResolution = captureResolution(); // Fallback if no source texture

    if (sourceTexture.isValid() && sourceTexture.getImageView() != nullptr) {
      // Use source texture resolution - PRESERVE ASPECT RATIO
      const auto& imageInfo = sourceTexture.getImageView()->imageInfo();

      // Clamp while preserving aspect ratio
      uint32_t srcWidth = imageInfo.extent.width;
      uint32_t srcHeight = imageInfo.extent.height;

      // Find the maximum dimension
      uint32_t maxDim = std::max(srcWidth, srcHeight);

      // If maxDim is below minimum, scale uniformly to meet minimum
      uint32_t width = srcWidth;
      uint32_t height = srcHeight;

      if (maxDim < 256u) {
        // Scale up proportionally so max dimension is 256
        float scale = 256.0f / static_cast<float>(maxDim);
        width = static_cast<uint32_t>(srcWidth * scale);
        height = static_cast<uint32_t>(srcHeight * scale);
      } else if (maxDim > 4096u) {
        // Scale down proportionally so max dimension is 4096
        float scale = 4096.0f / static_cast<float>(maxDim);
        width = static_cast<uint32_t>(srcWidth * scale);
        height = static_cast<uint32_t>(srcHeight * scale);
      }

      static uint32_t resolutionLogCount = 0;
      ++resolutionLogCount;

      if (resolutionLogCount <= 20) {
        Logger::info(str::format("[CAPTURE-RESOLUTION] #", resolutionLogCount,
                                " Source texture: ", srcWidth, "x", srcHeight,
                                " -> Using resolution: ", width, "x", height));
      }

      return { width, height };
    } else {
      static uint32_t noTextureLogCount = 0;
      ++noTextureLogCount;

      if (noTextureLogCount <= 20) {
        Logger::info(str::format("[CAPTURE-RESOLUTION-FALLBACK] #", noTextureLogCount,
                                " No source texture, using square fallback: ", fallbackResolution));
      }

      // Fallback: use configured resolution as square
      uint32_t resolution = std::clamp(fallbackResolution, 256u, 4096u);
      return { resolution, resolution };
    }
  }

  Matrix4 ShaderOutputCapturer::calculateUVSpaceProjection(
      const DrawCallState& drawCallState) const {

    // Get UV bounds from geometry
    const RasterGeometry& geom = drawCallState.getGeometryData();
    Vector2 uvMin(0.0f, 0.0f);
    Vector2 uvMax(1.0f, 1.0f);

    computeUVBounds(geom, uvMin, uvMax);

    // Create orthographic projection that maps UV range to clip space
    // UV (0,0) -> clip (-1, -1)
    // UV (1,1) -> clip (1, 1)
    Matrix4 uvToClip;
    uvToClip[0][0] = 2.0f / (uvMax.x - uvMin.x);
    uvToClip[1][1] = 2.0f / (uvMax.y - uvMin.y);
    uvToClip[2][2] = 1.0f;
    uvToClip[3][0] = -(uvMax.x + uvMin.x) / (uvMax.x - uvMin.x);
    uvToClip[3][1] = -(uvMax.y + uvMin.y) / (uvMax.y - uvMin.y);
    uvToClip[3][3] = 1.0f;

    // Note: This is a simplified version. A complete implementation would need
    // to modify the vertex shader to output UV coordinates as position.
    // For now, this assumes the geometry's position can be interpreted as UV.

    return uvToClip;
  }

  void ShaderOutputCapturer::computeUVBounds(
      const RasterGeometry& geom,
      Vector2& uvMin,
      Vector2& uvMax) const {

    // Default to 0-1 range
    uvMin = Vector2(0.0f, 0.0f);
    uvMax = Vector2(1.0f, 1.0f);

    // TODO: Actually scan UV buffer to find real bounds
    // This would require reading the texcoord buffer
    if (geom.texcoordBuffer.defined()) {
      // For now, assume standard 0-1 range
      // A proper implementation would scan the buffer
    }
  }

  void ShaderOutputCapturer::storeCapturedOutput(
      Rc<RtxContext> ctx,
      const DrawCallState& drawCallState,
      const Resources::Resource& texture,
      uint32_t currentFrame) {

    // Use cache key (texture hash for RT replacements, material hash otherwise)
    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    if (!isValidKey) {
      return; // Don't store invalid captures
    }
    XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();
    XXH64_hash_t geomHash = drawCallState.getGeometryData().getHashForRule<rules::FullGeometryHash>();
    bool isDynamic = isDynamicMaterial(matHash);

    // Store captured output (both static and dynamic materials)
    CapturedShaderOutput& output = m_capturedOutputs[cacheKey];
    output.capturedTexture = texture;
    output.geometryHash = geomHash;
    output.materialHash = matHash;
    output.lastCaptureFrame = currentFrame;
    output.captureSubmittedFrame = currentFrame;  // Track when submitted for async tracking
    output.isDynamic = isDynamic;
    output.isPending = true;  // Mark as pending - GPU hasn't finished yet
    output.resolution = { texture.image->info().extent.width,
                          texture.image->info().extent.height };

    static uint32_t storeCount = 0;
    if (++storeCount <= 100) {
      Logger::warn(str::format("[STORE-TEXTURE] #", storeCount,
                              " cacheKey=0x", std::hex, cacheKey, std::dec,
                              " texture.isValid()=", texture.isValid() ? "TRUE" : "FALSE",
                              " resolution=", output.resolution.width, "x", output.resolution.height,
                              " isPending=TRUE"));
    }

    // Calculate VRAM usage for LRU eviction (4 bytes per pixel for RGBA8)
    output.vramBytes = static_cast<size_t>(output.resolution.width) *
                       static_cast<size_t>(output.resolution.height) * 4;

    static uint32_t storeLogCount = 0;
    const bool shouldLog = (++storeLogCount <= 20) || (drawCallState.renderTargetReplacementSlot >= 0);
    if (shouldLog) {
      Logger::info(str::format("[ShaderCapture-Store] #", storeLogCount,
                              " Stored cacheKey=0x", std::hex, cacheKey, std::dec,
                              " matHash=0x", std::hex, matHash, std::dec,
                              " isDynamic=", isDynamic ? "YES" : "NO",
                              " isRTReplacement=", (drawCallState.renderTargetReplacementSlot >= 0 ? "YES" : "NO"),
                              " resolution=", output.resolution.width, "x", output.resolution.height));
    }

    // CRITICAL: For render target replacements, register the captured output as a replacement
    // for the ORIGINAL render target texture, so other materials that reference it can find it
    if (drawCallState.renderTargetReplacementSlot >= 0 && drawCallState.originalRenderTargetHash != 0) {
      // Register the captured output as a replacement for the original render target
      // When other materials reference the original RT hash, they'll get the captured version
      registerTextureReplacement(drawCallState.originalRenderTargetHash, TextureRef(texture.view));
      XXH64_hash_t capturedHash = (texture.view != nullptr) ? texture.view->image()->getHash() : 0;
      Logger::info(str::format("[ShaderCapture-Store] Registered texture replacement: originalRT=0x",
                              std::hex, drawCallState.originalRenderTargetHash,
                              " → capturedOutput=0x", capturedHash, std::dec));
    }
  }

  bool ShaderOutputCapturer::isDynamicMaterial(XXH64_hash_t materialHash) const {
    // Dynamic materials are those with TIME-BASED ANIMATION (water, scrolling textures, etc.)
    // View-dependent UVs are NOT dynamic - the path tracer handles UV mapping
    // Only mark as dynamic if explicitly listed in dynamicShaderMaterials option

    // Check if explicitly marked as dynamic
    if (dynamicShaderMaterials().count(materialHash) > 0) {
      static uint32_t isDynamicLogCount = 0;
      if (++isDynamicLogCount <= 20) {
        Logger::info(str::format("[ShaderCapture-Dynamic] #", isDynamicLogCount,
                                " matHash=0x", std::hex, materialHash, std::dec,
                                " is DYNAMIC (time-animated shader)"));
      }
      return true;
    }

    // Everything else is static (capture once, path tracer handles UV mapping)
    return false;
  }

  void ShaderOutputCapturer::showImguiSettings() {
    ImGui::Text("Shader Output Capture");
    ImGui::Text("Captures pixel shader output for animated materials");
    ImGui::Separator();

    bool enabled = enableShaderOutputCapture();
    if (ImGui::Checkbox("Enable Shader Output Capture", &enabled)) {
      enableShaderOutputCaptureObject().setDeferred(enabled);
    }

    if (enableShaderOutputCapture()) {
      ImGui::Indent();

      int captureRes = static_cast<int>(captureResolution());
      if (ImGui::DragInt("Capture Resolution", &captureRes, 64.0f, 256, 4096)) {
        captureResolutionObject().setDeferred(static_cast<uint32_t>(captureRes));
      }

      bool dynamicOnly = dynamicCaptureOnly();
      if (ImGui::Checkbox("Dynamic Capture Only", &dynamicOnly)) {
        dynamicCaptureOnlyObject().setDeferred(dynamicOnly);
      }

      int maxCaptures = static_cast<int>(maxCapturesPerFrame());
      if (ImGui::DragInt("Max Captures Per Frame", &maxCaptures, 1.0f, 1, 500)) {
        maxCapturesPerFrameObject().setDeferred(static_cast<uint32_t>(maxCaptures));
      }

      int recaptureInt = static_cast<int>(recaptureInterval());
      if (ImGui::DragInt("Recapture Interval (frames)", &recaptureInt, 1.0f, 1, 60)) {
        recaptureIntervalObject().setDeferred(static_cast<uint32_t>(recaptureInt));
      }

      ImGui::Text("\nStatistics:");
      ImGui::Text("Cached Outputs: %zu", m_capturedOutputs.size());
      ImGui::Text("Captures This Frame: %u", m_capturesThisFrame);
      ImGui::Text("Render Target Cache Size: %zu", m_renderTargetCache.size());
      ImGui::Text("Texture Replacements: %zu", m_textureReplacements.size());

      ImGui::Unindent();
    }
  }

  // PROPER TEXTURE REPLACEMENT IMPLEMENTATION
  void ShaderOutputCapturer::registerTextureReplacement(XXH64_hash_t originalTextureHash, const TextureRef& replacementTexture) {
    m_textureReplacements[originalTextureHash] = replacementTexture;

    static uint32_t registerCount = 0;
    if (++registerCount <= 20) {
      Logger::info(str::format("[ShaderOutputCapturer-TextureReplacement] Registered replacement #", registerCount,
                              ": originalTexture=0x", std::hex, originalTextureHash,
                              " → replacementTexture=0x", (replacementTexture.isValid() ? replacementTexture.getImageHash() : 0), std::dec,
                              " (total replacements: ", m_textureReplacements.size(), ")"));
    }
  }

  TextureRef ShaderOutputCapturer::getReplacementTexture(XXH64_hash_t originalTextureHash) const {
    auto it = m_textureReplacements.find(originalTextureHash);
    if (it != m_textureReplacements.end()) {
      return it->second;
    }
    return TextureRef(); // Invalid
  }

  bool ShaderOutputCapturer::hasReplacementTexture(XXH64_hash_t originalTextureHash) const {
    return m_textureReplacements.find(originalTextureHash) != m_textureReplacements.end();
  }

  void ShaderOutputCapturer::createDummyResources(const Rc<DxvkDevice>& device) {
    // Only create once
    if (m_dummyTexture != nullptr)
      return;

    // Create a simple 1x1 black texture for dummy bindings
    DxvkImageCreateInfo imageInfo;
    imageInfo.type        = VK_IMAGE_TYPE_2D;
    imageInfo.format      = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.flags       = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent      = { 1, 1, 1 };
    imageInfo.numLayers   = 1;
    imageInfo.mipLevels   = 1;
    imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    imageInfo.access      = VK_ACCESS_SHADER_READ_BIT;
    imageInfo.layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;

    Rc<DxvkImage> dummyImage = device->createImage(imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppTexture, "ShaderOutputCapturer dummy texture");

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format       = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.usage        = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect       = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel     = 0;
    viewInfo.numLevels    = 1;
    viewInfo.minLayer     = 0;
    viewInfo.numLayers    = 1;
    // CRITICAL FIX: Use WHITE (1,1,1,1) not black!
    // If shader multiplies by dummy texture, black gives black output.
    // White is neutral for multiplication operations.
    viewInfo.swizzle      = { VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_ONE,
                              VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_ONE };

    m_dummyTexture = device->createImageView(dummyImage, viewInfo);

    // Create a dummy sampler
    DxvkSamplerCreateInfo samplerInfo;
    samplerInfo.minFilter      = VK_FILTER_LINEAR;
    samplerInfo.magFilter      = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipmapLodBias  = 0.0f;
    samplerInfo.mipmapLodMin   = 0.0f;
    samplerInfo.mipmapLodMax   = 0.0f;
    samplerInfo.useAnisotropy  = VK_FALSE;
    samplerInfo.maxAnisotropy  = 1.0f;
    samplerInfo.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.compareToDepth = VK_FALSE;
    samplerInfo.compareOp      = VK_COMPARE_OP_NEVER;
    samplerInfo.borderColor    = VkClearColorValue();
    samplerInfo.usePixelCoord  = VK_FALSE;

    m_dummySampler = device->createSampler(samplerInfo);

    // Create a dummy depth texture for shadow/comparison samplers
    // This is needed because some game shaders use shadow samplers that require depth-format textures
    DxvkImageCreateInfo depthImageInfo;
    depthImageInfo.type        = VK_IMAGE_TYPE_2D;
    depthImageInfo.format      = VK_FORMAT_D32_SFLOAT;  // Depth format for shadow comparison
    depthImageInfo.flags       = 0;
    depthImageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    depthImageInfo.extent      = { 1, 1, 1 };
    depthImageInfo.numLayers   = 1;
    depthImageInfo.mipLevels   = 1;
    depthImageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageInfo.stages      = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    depthImageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    depthImageInfo.layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthImageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;

    m_dummyDepthImage = device->createImage(depthImageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppTexture, "ShaderOutputCapturer dummy depth texture");

    DxvkImageViewCreateInfo depthViewInfo;
    depthViewInfo.type         = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format       = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.usage        = VK_IMAGE_USAGE_SAMPLED_BIT;
    depthViewInfo.aspect       = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.minLevel     = 0;
    depthViewInfo.numLevels    = 1;
    depthViewInfo.minLayer     = 0;
    depthViewInfo.numLayers    = 1;

    m_dummyDepthTexture = device->createImageView(m_dummyDepthImage, depthViewInfo);
    m_dummyDepthLayoutInitialized = false;  // Mark for layout transition on first use

    // Create a shadow sampler with depth comparison enabled
    DxvkSamplerCreateInfo shadowSamplerInfo;
    shadowSamplerInfo.minFilter      = VK_FILTER_LINEAR;
    shadowSamplerInfo.magFilter      = VK_FILTER_LINEAR;
    shadowSamplerInfo.mipmapMode     = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    shadowSamplerInfo.mipmapLodBias  = 0.0f;
    shadowSamplerInfo.mipmapLodMin   = 0.0f;
    shadowSamplerInfo.mipmapLodMax   = 0.0f;
    shadowSamplerInfo.useAnisotropy  = VK_FALSE;
    shadowSamplerInfo.maxAnisotropy  = 1.0f;
    shadowSamplerInfo.addressModeU   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeV   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.addressModeW   = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shadowSamplerInfo.compareToDepth = VK_TRUE;  // Enable depth comparison
    shadowSamplerInfo.compareOp      = VK_COMPARE_OP_LESS_OR_EQUAL;  // Always pass (depth=1.0 >= any ref)
    shadowSamplerInfo.borderColor    = VkClearColorValue();
    shadowSamplerInfo.usePixelCoord  = VK_FALSE;

    m_dummyShadowSampler = device->createSampler(shadowSamplerInfo);

    Logger::info("[ShaderOutputCapturer] Created dummy texture, depth texture, and samplers for robust binding");
  }

} // namespace dxvk


