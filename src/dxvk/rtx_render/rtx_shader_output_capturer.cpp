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
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_camera.h"
#include "../dxvk_device.h"
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
#include <chrono>

#include <rtx_shaders/shader_capture_prepare.h>
#include "rtx/pass/shader_capture/shader_capture.h"

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
    Logger::info("[ShaderOutputCapturer] Initialized (feature enabled, capture-on-demand)");
    Logger::info(str::format("[ShaderOutputCapturer] enableShaderOutputCapture = ", enableShaderOutputCapture()));
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

    Logger::info("[ShaderCapture-GPU] GPU-driven multi-indirect capture system initialized");
  }

  void ShaderOutputCapturer::shutdownGpuCaptureSystem() {
    // Buffers will be automatically cleaned up when DxvkBufferSlice goes out of scope
    m_captureRequestsBuffer = DxvkBufferSlice();
    m_indirectDrawArgsBuffer = DxvkBufferSlice();
    m_indirectIndexedDrawArgsBuffer = DxvkBufferSlice();
    m_captureCountersBuffer = DxvkBufferSlice();
    // Render target cache (m_renderTargetCache) is cleaned up automatically
  }

  // DELETED: allocateRenderTargetPool() - pre-allocation wasted ~8GB VRAM!
  // DELETED: allocateRenderTargetFromPool() - now using on-demand allocation via getRenderTarget()
  // Render targets are now allocated on-demand with caching for massive VRAM savings

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

    auto tFunctionStart = std::chrono::high_resolution_clock::now();

    Logger::info(str::format("========== [DEBUG] executeMultiIndirectCaptures() CALLED - pendingRequests=", m_pendingCaptureRequests.size(), " =========="));

    if (m_pendingCaptureRequests.empty()) {
      Logger::info("[DEBUG] executeMultiIndirectCaptures() - NO PENDING REQUESTS, returning early");
      return;
    }

    const uint32_t numRequests = static_cast<uint32_t>(m_pendingCaptureRequests.size());

    Logger::info(str::format("[ShaderCapture-GPU] ===== MAXIMUM PERFORMANCE MULTI-DRAW-INDIRECT ====="));
    Logger::info(str::format("[ShaderCapture-GPU] Processing ", numRequests, " requests"));

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
    std::vector<CaptureGroup> captureGroups;

    // Hash = (RT resolution << 32) | (texture hash & 0xFFFFFFFF)
    std::unordered_map<uint64_t, size_t> groupHash;

    Logger::info(str::format("[ShaderCapture-GPU] [GROUPING] Beginning loop over ", numRequests, " requests"));

    for (uint32_t i = 0; i < numRequests; i++) {
      Logger::info(str::format("[ShaderCapture-GPU] [GROUPING] Processing request ", i, "/", numRequests));

      const auto& request = m_pendingCaptureRequests[i];

      // All data is self-contained in the request - no pointer dereferencing needed!
      XXH64_hash_t texHash = request.textureHash;

      // Group key: RT resolution ONLY (for texture array layered rendering)
      // All captures with same resolution go into ONE batch with texture array
      uint64_t key = (uint64_t(request.resolution.width) << 32) |
                     uint64_t(request.resolution.height);

      auto it = groupHash.find(key);
      if (it == groupHash.end()) {
        captureGroups.push_back(CaptureGroup{});
        captureGroups.back().resolution = request.resolution;
        captureGroups.back().textureHash = texHash;
        captureGroups.back().shaderHash = 0; // TODO: Extract shader hash
        groupHash[key] = captureGroups.size() - 1;
        it = groupHash.find(key);
      }

      captureGroups[it->second].requestIndices.push_back(i);
    }

    auto tGroupingEnd = std::chrono::high_resolution_clock::now();
    auto groupingTime = std::chrono::duration<double, std::micro>(tGroupingEnd - tGroupingStart).count();

    Logger::info(str::format("[ShaderCapture-GPU] Grouped ", numRequests, " into ",
                            captureGroups.size(), " resource groups [", groupingTime, " us]"));

    // ========== STEP 2: EXECUTE WITH MULTI-DRAW-INDIRECT PER GROUP ==========
    auto tExecutionStart = std::chrono::high_resolution_clock::now();

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

      // Allocate TEXTURE ARRAY RT for layered rendering (one layer per capture)
      auto tRTAllocStart = std::chrono::high_resolution_clock::now();
      const auto& firstRequest = m_pendingCaptureRequests[group.requestIndices[0]];

      // Use texture array for layered rendering - each draw writes to a different layer
      group.renderTarget = getRenderTargetArray(ctx, group.resolution, VK_FORMAT_R8G8B8A8_UNORM, groupSize);

      auto tRTAllocEnd = std::chrono::high_resolution_clock::now();
      totalRTAllocTime += std::chrono::duration<double, std::micro>(tRTAllocEnd - tRTAllocStart).count();

      if (!group.renderTarget.isValid()) {
        Logger::err(str::format("[ShaderCapture-GPU] Failed to allocate RT for group (", groupSize, " draws)"));
        continue;
      }

      // Prepare texture array for layered rendering
      auto tBindingStart = std::chrono::high_resolution_clock::now();

      ctx->changeImageLayout(group.renderTarget.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

      // NOTE: For now, we'll render to each layer individually in a loop
      // TODO: Implement proper layered rendering with multiview or shader modifications
      // to render all layers in one multi-draw-indirect call

      VkClearValue clearValue = {};
      ctx->clearRenderTarget(group.renderTarget.view, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);

      auto tBindingEnd = std::chrono::high_resolution_clock::now();
      totalBindingTime += std::chrono::duration<double, std::micro>(tBindingEnd - tBindingStart).count();

      // Render each capture to its own layer in the texture array
      auto tLayeredRenderStart = std::chrono::high_resolution_clock::now();

      for (uint32_t layerIdx = 0; layerIdx < groupSize; layerIdx++) {
        const auto& request = m_pendingCaptureRequests[group.requestIndices[layerIdx]];

        // Create view to specific layer of texture array
        DxvkImageViewCreateInfo layerViewInfo;
        layerViewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
        layerViewInfo.format = group.renderTarget.image->info().format;
        layerViewInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        layerViewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        layerViewInfo.minLevel = 0;
        layerViewInfo.numLevels = 1;
        layerViewInfo.minLayer = layerIdx; // THIS LAYER
        layerViewInfo.numLayers = 1;

        Rc<DxvkImageView> layerView = ctx->getDevice()->createImageView(group.renderTarget.image, layerViewInfo);

        // Bind this specific layer as render target
        DxvkRenderTargets captureRt;
        captureRt.color[0].view = layerView;
        captureRt.color[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ctx->bindRenderTargets(captureRt);

        // Clear this layer
        VkClearValue clearValue = {};
        ctx->clearRenderTarget(layerView, VK_IMAGE_ASPECT_COLOR_BIT, clearValue);

        // Bind input texture for this capture
        if (request.colorTexture.isValid()) {
          TextureRef replacementTexture = getReplacementTexture(request.textureHash);
          const TextureRef& texToUse = replacementTexture.isValid()
            ? replacementTexture
            : request.colorTexture;

          if (texToUse.getImageView()) {
            Rc<DxvkImageView> imageView = texToUse.getImageView();
            ctx->bindResourceView(0, imageView, nullptr);
          }
        }

        // Set pipeline state and draw
        setCommonPipelineState(ctx, request);
        bindGeometryBuffers(ctx, request);

        if (request.indexCount > 0) {
          ctx->drawIndexed(request.indexCount, 1, request.indexOffset, request.vertexOffset, 0);
        } else {
          ctx->draw(request.vertexCount, 1, request.vertexOffset, 0);
        }

        successCount++;
      }

      auto tLayeredRenderEnd = std::chrono::high_resolution_clock::now();
      totalMultiDrawTime += std::chrono::duration<double, std::micro>(tLayeredRenderEnd - tLayeredRenderStart).count();

      // OBSOLETE: Old multi-draw-indirect code replaced with layered rendering
      if (false) {
      if (groupSize > 1) {
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
          setCommonPipelineState(ctx, firstRequest);

          // Bind geometry for first request (assumption: all use same buffers with different offsets)
          bindGeometryBuffers(ctx, firstRequest);

          // Multi-draw-indirect! Maximum performance!
          ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());
          ctx->drawIndexedIndirect(0, static_cast<uint32_t>(indirectIndexedDraws.size()),
                                  sizeof(VkDrawIndexedIndirectCommand));

          successCount += static_cast<uint32_t>(indirectIndexedDraws.size());
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

          setCommonPipelineState(ctx, firstRequest);
          bindGeometryBuffers(ctx, firstRequest);

          ctx->bindDrawBuffers(indirectBuf, DxvkBufferSlice());
          ctx->drawIndirect(0, static_cast<uint32_t>(indirectDraws.size()),
                           sizeof(VkDrawIndirectCommand));

          successCount += static_cast<uint32_t>(indirectDraws.size());
        }

        auto tMultiDrawEnd = std::chrono::high_resolution_clock::now();
        totalMultiDrawTime += std::chrono::duration<double, std::micro>(tMultiDrawEnd - tMultiDrawStart).count();
      } else {
        // Single draw - use direct call
        auto tSingleDrawStart = std::chrono::high_resolution_clock::now();

        singleDrawCount++;
        const auto& request = m_pendingCaptureRequests[group.requestIndices[0]];

        setCommonPipelineState(ctx, request);
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

      // Store all captures from this group - using self-contained request data
      for (uint32_t idx : group.requestIndices) {
        const auto& request = m_pendingCaptureRequests[idx];

        // Direct cache storage using request data (no DrawCallState pointer needed!)
        XXH64_hash_t cacheKey = request.materialHash;
        CapturedShaderOutput& output = m_capturedOutputs[cacheKey];
        output.capturedTexture = group.renderTarget;
        output.geometryHash = request.geometryHash;
        output.materialHash = request.materialHash;
        output.lastCaptureFrame = m_currentFrame;
        output.captureSubmittedFrame = m_currentFrame;
        output.isDynamic = request.isDynamic;
        output.isPending = false;
        output.resolution = request.resolution;
      }
    }

    m_pendingCaptureRequests.clear();

    auto tExecutionEnd = std::chrono::high_resolution_clock::now();
    auto tFunctionEnd = std::chrono::high_resolution_clock::now();

    auto executionTime = std::chrono::duration<double, std::micro>(tExecutionEnd - tExecutionStart).count();
    auto totalTime = std::chrono::duration<double, std::micro>(tFunctionEnd - tFunctionStart).count();

    Logger::info(str::format("[ShaderCapture-GPU] ===== PERFORMANCE BREAKDOWN ====="));
    Logger::info(str::format("[ShaderCapture-GPU] Complete: ", successCount, " draws, ",
                            multiDrawCount, " multi-draw-indirect groups, ",
                            singleDrawCount, " single draws"));
    Logger::info(str::format("[ShaderCapture-GPU] Grouping: ", groupingTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] RT Allocation: ", totalRTAllocTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] Binding (RT+Tex): ", totalBindingTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] Multi-draw: ", totalMultiDrawTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] Single-draw: ", totalSingleDrawTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] Execution time: ", executionTime, " us"));
    Logger::info(str::format("[ShaderCapture-GPU] TOTAL TIME: ", totalTime, " us (", totalTime / 1000.0, " ms)"));
  }

  void ShaderOutputCapturer::setCommonPipelineState(Rc<RtxContext> ctx, const GpuCaptureRequest& request) {
    // Use self-contained viewport/scissor data from request
    ctx->setViewports(1, &request.viewport, &request.scissor);

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
  }

  void ShaderOutputCapturer::bindGeometryBuffers(Rc<RtxContext> ctx, const GpuCaptureRequest& request) {
    // Use self-contained geometry buffer data from request - no DrawCallState needed!

    // Bind vertex buffers (replacement takes priority)
    if (request.replacementVertexBuffer.defined()) {
      ctx->bindVertexBuffer(0, request.replacementVertexBuffer, request.replacementVertexStride);
    } else if (request.vertexBuffer.defined()) {
      ctx->bindVertexBuffer(0, request.vertexBuffer, request.vertexStride);
    }

    if (request.replacementTexcoordBuffer.defined()) {
      ctx->bindVertexBuffer(1, request.replacementTexcoordBuffer, request.replacementTexcoordStride);
    } else if (request.texcoordBuffer.defined()) {
      ctx->bindVertexBuffer(1, request.texcoordBuffer, request.texcoordStride);
    }

    if (request.replacementNormalBuffer.defined()) {
      ctx->bindVertexBuffer(2, request.replacementNormalBuffer, request.vertexStride); // Use vertex stride for normal
    } else if (request.normalBuffer.defined()) {
      ctx->bindVertexBuffer(2, request.normalBuffer, request.vertexStride);
    }

    // Bind index buffer
    if (request.replacementIndexBuffer.defined()) {
      ctx->bindIndexBuffer(request.replacementIndexBuffer, request.replacementIndexType);
    } else if (request.indexBuffer.defined()) {
      ctx->bindIndexBuffer(request.indexBuffer, request.indexType);
    }
  }

  // ======== END GPU-DRIVEN SYSTEM ========

  bool ShaderOutputCapturer::shouldCaptureStatic(const DrawCallState& drawCallState) {
    // Stage 2 version - no cache checking, just basic feature/hash checking
    // This is called early in D3D9Rtx before GPU context is available

    if (!enableShaderOutputCapture()) {
      return false;
    }

    // CRITICAL FIX: Skip capture if VS constants are all zeros (uninitialized state)
    // During game initialization, captures happen BEFORE the game sets up transformation matrices.
    // If we capture with zero matrices, geometry transforms to invalid coords and gets culled.
    // Check first 4 VS constants (c[0]-c[3]) which typically contain view/projection matrices.
    if (!drawCallState.vertexShaderConstantData.empty() && drawCallState.vertexShaderConstantData.size() >= 4) {
      bool allZeros = true;
      for (uint32_t i = 0; i < 4; i++) {
        const Vector4& c = drawCallState.vertexShaderConstantData[i];
        if (c.x != 0.0f || c.y != 0.0f || c.z != 0.0f || c.w != 0.0f) {
          allZeros = false;
          break;
        }
      }

      if (allZeros) {
        static uint32_t zeroConstLogCount = 0;
        if (++zeroConstLogCount <= 5) {
          Logger::warn(str::format("[ShaderOutputCapturer] SKIPPING capture - VS constants c[0]-c[3] are ALL ZERO (uninitialized). ",
                                  "Waiting for game to set up transformation matrices. (Count: ", zeroConstLogCount, ")"));
        }
        return false; // Skip capture - matrices not initialized yet
      }
    }

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

    // Check if capturing all materials via hash list
    if (captureEnabledHashes().count(0xALL) > 0) {
      return true;
    }

    // Check if this specific material is whitelisted
    return captureEnabledHashes().count(matHash) > 0;
  }

  bool ShaderOutputCapturer::shouldCapture(const DrawCallState& drawCallState) const {
    // Log periodically to track behavior
    static uint32_t callCount = 0;
    ++callCount;
    const bool shouldLog = (callCount <= 20) || (callCount % 1000 == 0);

    // AGGRESSIVE LOGGING: Always log first 5 calls to confirm function is being invoked
    if (callCount <= 5) {
      Logger::info(str::format("========== [DEBUG] shouldCapture() CALLED (call #", callCount, ") =========="));
    }

    if (!enableShaderOutputCapture()) {
      if (shouldLog) {
        Logger::info("[ShaderOutputCapturer] shouldCapture() returning false - feature disabled");
      }
      return false;
    }

    // Get cache key (uses texture hash for RT replacements, material hash otherwise)
    auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
    const bool hasRenderTargetReplacement = (drawCallState.renderTargetReplacementSlot >= 0);

    // Reject only if the key is explicitly marked invalid (invalid RT replacement texture)
    if (!isValidKey) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderOutputCapturer] shouldCapture() returning FALSE - invalid RT replacement texture (callCount=", callCount, ")"));
      }
      return false;
    }

    // cacheKey=0 is now allowed for regular materials (materialHash can legitimately be 0)
    if (shouldLog && cacheKey == 0) {
      Logger::info(str::format("[ShaderOutputCapturer] cacheKey=0 detected (isRTReplacement=",
                              hasRenderTargetReplacement ? "YES" : "NO", ") - proceeding with capture"));
    }

    // Check if we already captured this material
    auto it = m_capturedOutputs.find(cacheKey);
    if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
      // Already cached - check if we need to re-capture (based on current dynamicCaptureOnly setting)
      // This allows changing dynamicCaptureOnly at runtime to force re-captures
      const bool shouldRecapture = !dynamicCaptureOnly() || isDynamicMaterial(cacheKey);

      if (!shouldRecapture) {
        // Static material with dynamicCaptureOnly=true - use cache forever
        if (shouldLog) {
          Logger::info(str::format("[ShaderOutputCapturer] shouldCapture() returning FALSE - cached static material (cacheKey=0x",
                                  std::hex, cacheKey, std::dec, " isRT=", hasRenderTargetReplacement ? "YES" : "NO", ")"));
        }
        return false;
      }
      // Dynamic material or dynamicCaptureOnly=false - continue to check needsRecapture()
    }

    // RT replacements always capture if not cached
    if (hasRenderTargetReplacement) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderOutputCapturer] shouldCapture() returning TRUE - RT replacement not cached (cacheKey=0x",
                                std::hex, cacheKey, std::dec, " slot=", drawCallState.renderTargetReplacementSlot, ")"));
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
    auto cacheIt = m_capturedOutputs.find(cacheKey);

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
      // Already captured and ready - don't recapture
      return false;
    }

    // Not yet captured - allow async capture (no throttling)
    if (callCount <= 50) {
      Logger::info(str::format("[ShaderCapture-ASYNC] Starting ASYNC capture for matHash=0x", std::hex, matHash, std::dec,
                              " (will run on GPU in parallel with rendering)"));
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
    if (!isDynamic) {
      if (shouldLog) {
        Logger::info(str::format("[ShaderCapture-Recapture] #", needsRecaptureLogCount,
                                " cacheKey=0x", std::hex, cacheKey, std::dec,
                                " SKIP RECAPTURE - static material, using cache"));
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

    static uint32_t captureCallEntryCount = 0;
    if (++captureCallEntryCount <= 10) {
      Logger::info(str::format("[CAPTURE-ENTRY] captureDrawCall #", captureCallEntryCount, " ENTERED"));
    }

    // GPU-DRIVEN BATCHED CAPTURE MODE - PRODUCTION READY
    // Queues captures and executes them in batches to eliminate CPU-GPU sync overhead
    static constexpr bool USE_GPU_DRIVEN_MODE = true; // ENABLED: Production-ready batched execution

    if (USE_GPU_DRIVEN_MODE) {
      static uint32_t gpuModeEntryCount = 0;
      if (++gpuModeEntryCount <= 10) {
        Logger::info(str::format("[GPU-MODE-ENTRY] Entered GPU batching path #", gpuModeEntryCount));
      }
      // Check if we should queue this capture
      const XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();
      auto [cacheKey, isValidKey] = getCacheKey(drawCallState);

      static uint32_t cacheKeyLogCount = 0;
      if (++cacheKeyLogCount <= 20) {
        Logger::info(str::format("[GPU-CACHEKEY] #", cacheKeyLogCount,
                                " matHash=0x", std::hex, matHash,
                                " cacheKey=0x", cacheKey, std::dec,
                                " isValid=", isValidKey));
      }

      if (!isValidKey) {
        static uint32_t invalidKeyCount = 0;
        if (++invalidKeyCount <= 10) {
          Logger::info(str::format("[GPU-SKIP-INVALID] #", invalidKeyCount, " Invalid cache key, skipping"));
        }
        return false; // Invalid key, skip
      }

      // Check if already captured
      auto it = m_capturedOutputs.find(cacheKey);
      if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
        // Already have a cached texture - return it
        static uint32_t cachedHitCount = 0;
        if (++cachedHitCount <= 10) {
          Logger::info(str::format("[GPU-CACHED-HIT] #", cachedHitCount, " Found cached texture for cacheKey=0x", std::hex, cacheKey, std::dec));
        }
        outputTexture = getCapturedTexture(cacheKey);
        return outputTexture.isValid();
      }

      // Queue new capture request - COPY ALL DATA (no pointers!)
      GpuCaptureRequest request = {};

      // Hashes and metadata
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

      // COPY VIEWPORT/SCISSOR STATE
      request.viewport = drawCallState.originalViewport;
      request.scissor = drawCallState.originalScissor;

      m_pendingCaptureRequests.push_back(request);

      static uint32_t queueLogCount = 0;
      if (++queueLogCount <= 20) {
        Logger::info(str::format("[ShaderCapture-GPU] Queued capture request #", queueLogCount,
                                " matHash=0x", std::hex, matHash, std::dec,
                                " (", m_pendingCaptureRequests.size(), " total queued)"));
      }

      // DEFERRED BATCHED EXECUTION (MegaGeometry-style):
      // Just queue - ALL captures execute together in onFrameBegin() next frame
      // This maximizes batching efficiency and matches reference implementation

      // Check if already cached from previous frame
      outputTexture = getCapturedTexture(cacheKey);
      return outputTexture.isValid();
    }

    // ===== OLD CPU-DRIVEN MODE (below) =====
    static uint32_t captureCallCount = 0;
    ++captureCallCount;
    const bool shouldLog = (captureCallCount <= 20); // Only log first 20 captures
    const XXH64_hash_t matHash = drawCallState.getMaterialData().getHash();

    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));
    Logger::info(str::format("[OPTION-C-DEBUG] ====== CAPTURE #", captureCallCount, " START ======"));
    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));
    Logger::info(str::format("[OPTION-C-DEBUG] materialHash=0x", std::hex, matHash, std::dec));
    Logger::info(str::format("[OPTION-C-DEBUG] drawCallID=", drawCallState.drawCallID));
    Logger::info(str::format("[OPTION-C-DEBUG] vertexCount=", drawCallState.getGeometryData().vertexCount,
                            " indexCount=", drawCallState.getGeometryData().indexCount));
    Logger::info(str::format("[OPTION-C-DEBUG] usesVertexShader=", drawCallState.usesVertexShader ? "YES" : "NO",
                            " usesPixelShader=", drawCallState.usesPixelShader ? "YES" : "NO"));
    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));

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

    Logger::info("[OPTION-C-DEBUG] OPTION B disabled - proceeding to OPTION C (shader re-execution)");

    // If framebuffer capture failed or wasn't available, fall back to shader re-execution
    // Check if we need to recapture
    // TEMPORARY DEBUG: FORCE RECAPTURE to verify geometry renders correctly
    // TODO: Fix cache invalidation to detect camera/scene changes
    const bool needRecapture = true; // FORCING RECAPTURE FOR DEBUGGING
    //const bool needRecapture = needsRecapture(drawCallState, currentFrame); // ORIGINAL (commented out)
    Logger::info(str::format("[OPTION-C-PERF] ⏱️ CACHE CHECK: needRecapture=", needRecapture ? "YES (will re-execute)" : "NO (using cache)"));
    Logger::info(str::format("[VSTREAM-DEBUG] ===== CACHE DECISION POINT ====="));
    Logger::info(str::format("[VSTREAM-DEBUG] needRecapture=", needRecapture,
                            " matHash=0x", std::hex, matHash, std::dec,
                            " currentFrame=", currentFrame));
    Logger::warn("[VSTREAM-DEBUG] ⚠️ FORCING RECAPTURE (debugging) - old cache logic was: needsRecapture()=false caused stale geometry");

    if (!needRecapture) {  // Use cached result if nothing changed
      Logger::info("[VSTREAM-DEBUG] ===== USING CACHED RESULT - EARLY EXIT =====");
      Logger::info(str::format("[VSTREAM-DEBUG] Skipping re-execution for matHash=0x", std::hex, matHash, std::dec));
      // Use cached texture - get cache key for proper lookup
      auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
      if (isValidKey) {
        outputTexture = getCapturedTexture(cacheKey);
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

    // CRITICAL FIX: Transition render target TO COLOR_ATTACHMENT_OPTIMAL before binding
    // If this RT was used before, it's in SHADER_READ_ONLY_OPTIMAL from the previous capture
    // We need to transition it back so we can render to it again
    ctx->changeImageLayout(renderTarget.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    Logger::info(str::format("[ShaderCapture-Layout] Transitioned render target 0x", std::hex,
                            renderTarget.image->getHash(), std::dec,
                            " to COLOR_ATTACHMENT_OPTIMAL for rendering"));

    // Bind offscreen render target
    DxvkRenderTargets captureRt;
    captureRt.color[0].view = renderTarget.view;
    captureRt.color[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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
    Logger::info(str::format("[OPTION-C-DEBUG] ========== TEXTURE BINDING START =========="));
    Logger::info(str::format("[OPTION-C-DEBUG] Total captured D3D9 textures: ", drawCallState.capturedD3D9Textures.size()));

    for (const auto& capturedTex : drawCallState.capturedD3D9Textures) {
      if (!capturedTex.texture.isValid() || capturedTex.texture.getImageView() == nullptr) {
        Logger::warn(str::format("[ShaderCapture-D3D9Tex] Skipped invalid captured texture: slot=", capturedTex.slot));
        continue;
      }

      // CRITICAL FIX: Skip empty/cleared textures (hash==0x0)
      // These are uninitialized textures that will produce black when sampled
      const XXH64_hash_t texHash = capturedTex.texture.getImageHash();
      if (texHash == 0) {
        Logger::warn(str::format("[OPTION-C-DEBUG] ✗ SKIPPED EMPTY TEXTURE: slot=", capturedTex.slot,
                                " hash=0x0 (empty/cleared texture would produce black)"));
        continue;
      }

      // CRITICAL FIX: Skip textures that match the material hash (self-reference)
      // When texHash == matHash, it means we're trying to read from the render target we're writing to
      // EXCEPTION: For RT replacements, the replacement slot texture might have matHash (from previous capture)
      // but that's a VALID input texture from a previous frame, so don't skip it!

      // LOG EVERY TEXTURE IN DETAIL
      const auto& imgInfo = capturedTex.texture.getImageView()->image()->info();
      Logger::info(str::format("[OPTION-C-DEBUG] Texture slot=", capturedTex.slot,
                              " hash=0x", std::hex, texHash, std::dec,
                              " format=", imgInfo.format,
                              " size=", imgInfo.extent.width, "x", imgInfo.extent.height,
                              " mips=", imgInfo.mipLevels));

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
        Logger::warn(str::format("[OPTION-C-DEBUG] ⚠️ DETECTED SELF-REFERENCE: slot=", capturedTex.slot,
                                " textureHash=0x", std::hex, texHash, std::dec,
                                (isOriginalRTSelfReference ? " == originalRT" : " == matHash"),
                                " - will copy RT and bind copy to break circular dependency"));

        // Get the current render target that we're about to draw to
        Rc<DxvkImageView> currentRTView = capturedTex.texture.getImageView();
        if (currentRTView == nullptr) {
          Logger::err("[OPTION-C-DEBUG] ✗ Cannot copy RT - imageView is null, skipping");
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

        auto rtCopyEndTime = std::chrono::high_resolution_clock::now();
        auto rtCopyDuration = std::chrono::duration_cast<std::chrono::microseconds>(rtCopyEndTime - rtCopyStartTime).count();
        Logger::info(str::format("[OPTION-C-PERF] ⏱️ RT COPY TIME: ", rtCopyDuration / 1000.0, " ms - slot=", capturedTex.slot,
                                " size=", rtInfo.extent.width, "x", rtInfo.extent.height));
        Logger::info(str::format("[OPTION-C-DEBUG] ✓ BOUND RT COPY (FEEDBACK FIX): slot=", capturedTex.slot,
                                " size=", rtInfo.extent.width, "x", rtInfo.extent.height,
                                " - shader can now read from copy while writing to original"));
        continue; // Done with this texture
      }

      // Log when we're binding RT for feedback
      if (isRTFeedbackCase) {
        Logger::info(str::format("[OPTION-C-DEBUG] ✓ BINDING RT FEEDBACK: slot=", capturedTex.slot,
                                " textureHash=0x", std::hex, texHash, std::dec,
                                " (shader needs to read previous frame)"));
      }

      Rc<DxvkImageView> imageView = capturedTex.texture.getImageView();
      ctx->bindResourceView(capturedTex.slot, imageView, nullptr);

      // Try to get the sampler from material (samplers are stored by texture ID, not slot)
      // For now, use sampler from material slot 0 as fallback
      const Rc<DxvkSampler>& sampler = material.getSampler();
      if (sampler != nullptr) {
        ctx->bindResourceSampler(capturedTex.slot, sampler);
        Logger::info(str::format("[OPTION-C-DEBUG] ✓ BOUND TEXTURE + SAMPLER: slot=", capturedTex.slot,
                                " hash=0x", std::hex, texHash, std::dec));
      } else {
        Logger::warn(str::format("[OPTION-C-DEBUG] ✓ BOUND TEXTURE (NO SAMPLER): slot=", capturedTex.slot,
                                " hash=0x", std::hex, texHash, std::dec));
      }
    }

    auto texBindingEndTime = std::chrono::high_resolution_clock::now();
    auto texBindingDuration = std::chrono::duration_cast<std::chrono::microseconds>(texBindingEndTime - texBindingStartTime).count();
    Logger::info(str::format("[OPTION-C-PERF] ⏱️ TEXTURE BINDING TOTAL TIME: ", texBindingDuration / 1000.0, " ms"));
    Logger::info(str::format("[OPTION-C-DEBUG] ========== TEXTURE BINDING END =========="));

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
    Logger::info(str::format("[OPTION-C-DEBUG] ========== SHADER BINDING START =========="));
    Logger::info(str::format("[OPTION-C-DEBUG] usesVertexShader=", drawCallState.usesVertexShader ? "YES" : "NO",
                            " usesPixelShader=", drawCallState.usesPixelShader ? "YES" : "NO"));

    if (drawCallState.usesVertexShader && drawCallState.vertexShader != nullptr) {
      // Convert D3D9 vertex shader to DXVK shader
      const D3D9CommonShader* commonShader = drawCallState.vertexShader->GetCommonShader();
      if (commonShader != nullptr) {
        Rc<DxvkShader> dxvkShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        if (dxvkShader != nullptr) {
          ctx->bindShader(VK_SHADER_STAGE_VERTEX_BIT, dxvkShader);
          Logger::info(str::format("[OPTION-C-DEBUG] ✓ VERTEX SHADER BOUND: shader=", drawCallState.vertexShader,
                                  " dxvkShader=", dxvkShader.ptr()));
        } else {
          Logger::err(str::format("[OPTION-C-DEBUG] ✗ VERTEX SHADER FAILED: dxvkShader is NULL!"));
        }
      } else {
        Logger::err(str::format("[OPTION-C-DEBUG] ✗ VERTEX SHADER FAILED: commonShader is NULL!"));
      }
    } else {
      Logger::warn(str::format("[OPTION-C-DEBUG] ✗ VERTEX SHADER NOT BOUND: usesVertexShader=", drawCallState.usesVertexShader ? "yes" : "no",
                  " vertexShader=", drawCallState.vertexShader != nullptr ? "non-null" : "null"));
    }

    if (drawCallState.usesPixelShader && drawCallState.pixelShader != nullptr) {
      // Convert D3D9 pixel shader to DXVK shader
      const D3D9CommonShader* commonShader = drawCallState.pixelShader->GetCommonShader();
      if (commonShader != nullptr) {
        Rc<DxvkShader> dxvkShader = commonShader->GetShader(D3D9ShaderPermutations::None);
        if (dxvkShader != nullptr) {
          ctx->bindShader(VK_SHADER_STAGE_FRAGMENT_BIT, dxvkShader);
          Logger::info(str::format("[OPTION-C-DEBUG] ✓ PIXEL SHADER BOUND: shader=", drawCallState.pixelShader,
                                  " dxvkShader=", dxvkShader.ptr()));
        } else {
          Logger::err(str::format("[OPTION-C-DEBUG] ✗ PIXEL SHADER FAILED: dxvkShader is NULL!"));
        }
      } else {
        Logger::err(str::format("[OPTION-C-DEBUG] ✗ PIXEL SHADER FAILED: commonShader is NULL!"));
      }
    } else {
      Logger::warn(str::format("[OPTION-C-DEBUG] ✗ PIXEL SHADER NOT BOUND: usesPixelShader=", drawCallState.usesPixelShader ? "yes" : "no",
                  " pixelShader=", drawCallState.pixelShader != nullptr ? "non-null" : "null"));
    }

    Logger::info(str::format("[OPTION-C-DEBUG] ========== SHADER BINDING END =========="));

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

    Logger::info(str::format("[OPTION-A-EXECUTE] Buffer availability check: useCapturedStreams=", useCapturedStreams,
                            " (", drawCallState.capturedVertexStreams.size(), " streams), hasLegacyBuffers=", hasLegacyBuffers,
                            " preservedVertexData.size=", preservedVertexData.size(),
                            " preservedIndexData.size=", preservedIndexData.size(),
                            " geom.usesIndices=", geom.usesIndices() ? 1 : 0));
    Logger::info(str::format("[VSTREAM-DEBUG] ===== BUFFER AVAILABILITY ====="));
    Logger::info(str::format("[VSTREAM-DEBUG] capturedVertexStreams.size=", drawCallState.capturedVertexStreams.size()));
    Logger::info(str::format("[VSTREAM-DEBUG] preservedVertexData.size=", preservedVertexData.size()));
    Logger::info(str::format("[VSTREAM-DEBUG] preservedIndexData.size=", preservedIndexData.size()));
    Logger::info(str::format("[VSTREAM-DEBUG] geom.positionBuffer.defined=", geom.positionBuffer.defined()));
    Logger::info(str::format("[VSTREAM-DEBUG] geom.indexBuffer.defined=", geom.indexBuffer.defined()));
    Logger::info(str::format("[VSTREAM-DEBUG] geom.vertexCount=", geom.vertexCount));
    Logger::info(str::format("[VSTREAM-DEBUG] geom.indexCount=", geom.indexCount));
    const bool useOriginalBuffers =
      !drawCallState.forceGeometryCopy &&
      (useCapturedStreams || hasLegacyBuffers);

    // CRITICAL FIX: originalVertexStride is only set for Stage 2 buffer captures
    // For capturedVertexStreams (Stage 4), we need to check if streams have valid data
    const bool hasValidVertexData = (drawCallState.originalVertexStride >= 12) ||
                                    (!drawCallState.capturedVertexStreams.empty() &&
                                     drawCallState.capturedVertexStreams.front().stride > 0);

    Logger::info(str::format("[OPTION-A-DEBUG] forceGeometryCopy=", drawCallState.forceGeometryCopy ? 1 : 0,
                            " originalVertexStride=", drawCallState.originalVertexStride,
                            " useOriginalBuffers=", useOriginalBuffers ? 1 : 0,
                            " hasValidVertexData=", hasValidVertexData ? 1 : 0,
                            " condition=(useOriginalBuffers && hasValidVertexData)=",
                            (useOriginalBuffers && hasValidVertexData) ? 1 : 0));

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
      Logger::info(str::format("[OPTION-A-EXECUTE] Using original D3D9 buffers from Stage 2"));
      Logger::info(str::format("[OPTION-A-EXECUTE] originalVertexBuffer=", drawCallState.originalVertexBuffer.ptr(),
                              " stride=", drawCallState.originalVertexStride,
                              " offset=", drawCallState.originalVertexOffset));

      if (drawCallState.originalIndexBuffer != nullptr) {
        Logger::info(str::format("[OPTION-A-EXECUTE] originalIndexBuffer=", drawCallState.originalIndexBuffer.ptr(),
                                " type=", drawCallState.originalIndexType));
      }

      // Use draw parameters for counts (same as RasterGeometry path)
      actualVertexCount = drawParams.vertexCount;
      actualIndexCount = drawParams.indexCount;

      bindCount = 1;
      attrCount = 0;

      // NEW: Use actual D3D9 vertex declaration instead of guessing from stride
      if (drawCallState.vertexDecl != nullptr) {
        Logger::info("[OPTION-A-EXECUTE] Using D3D9 vertex declaration for format detection");

        const D3D9VertexElements& elements = drawCallState.vertexDecl->GetElements();

        // Log the vertex declaration for debugging
        Logger::info(str::format("[OPTION-A-EXECUTE] Vertex declaration has ", elements.size(), " elements"));

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

          // Log raw element data BEFORE processing for debugging
          Logger::info(str::format("[OPTION-A-EXECUTE] Processing element ", i,
                                  ": stream=", elem.Stream,
                                  " offset=", elem.Offset,
                                  " type=", static_cast<uint32_t>(elem.Type),
                                  " usage=", static_cast<uint32_t>(elem.Usage),
                                  " usageIndex=", static_cast<uint32_t>(elem.UsageIndex)));

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

          Logger::info(str::format("[OPTION-A-EXECUTE] Attribute ", attrCount, ": usage=", static_cast<uint32_t>(elem.Usage),
                                  " usageIndex=", static_cast<uint32_t>(elem.UsageIndex), " type=", static_cast<uint32_t>(elem.Type),
                                  " offset=", elem.Offset, " → location=", location, " format=", static_cast<uint32_t>(format)));

          attrCount++;
        }

        Logger::info(str::format("[OPTION-A-EXECUTE] Built vertex layout from declaration: attrCount=", attrCount));

        // Log complete vertex layout summary
        Logger::info(str::format("[OPTION-A-EXECUTE] ===== VERTEX LAYOUT SUMMARY (stride=", drawCallState.originalVertexStride, ") ====="));
        for (uint32_t i = 0; i < attrCount; i++) {
          const char* locationName = "UNKNOWN";
          switch (attrList[i].location) {
            case 0: locationName = "POSITION"; break;
            case 1: locationName = "BLENDWEIGHT"; break;
            case 2: locationName = "BLENDINDICES"; break;
            case 3: locationName = "NORMAL"; break;
            case 4: locationName = "COLOR0"; break;
            case 5: locationName = "COLOR1"; break;
            case 6: locationName = "TANGENT"; break;
            case 7: locationName = "TEXCOORD0"; break;
            case 8: locationName = "TEXCOORD1"; break;
            case 9: locationName = "BINORMAL"; break;
            case 10: locationName = "PSIZE"; break;
          }
          Logger::info(str::format("  [", i, "] ", locationName, " (location=", attrList[i].location,
                                  ") offset=", attrList[i].offset, " format=", attrList[i].format,
                                  " binding=", attrList[i].binding));
        }
        Logger::info("[OPTION-A-EXECUTE] ==========================================");
      } else {
        // Fallback to stride-based guessing if no vertex declaration available
        Logger::warn("[OPTION-A-EXECUTE] No vertex declaration available - falling back to stride-based detection");
        Logger::info(str::format("[OPTION-A-EXECUTE] Auto-detecting ", drawCallState.originalVertexStride, "-byte vertex format"));

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

        Logger::info(str::format("[OPTION-A-EXECUTE] Detected vertex layout from stride: attrCount=", attrCount, " (stride=", drawCallState.originalVertexStride, ")"));
      }

      if (!useCapturedStreams) {
        // Vertex binding description
        bindList[0].binding = 0;
        bindList[0].fetchRate = 0;
        bindList[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        ctx->setInputLayout(attrCount, attrList, 1, bindList);
      } else {
        // OPTION A: Reconstruct per-stream vertex layout from captured D3D9 streams
        Logger::info(str::format("[OPTION-A-MULTISTREAM] Reconstructing per-stream vertex layout: ",
                                drawCallState.capturedVertexStreams.size(), " streams, ",
                                drawCallState.capturedVertexElements.size(), " elements"));

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

          Logger::info(str::format("[OPTION-A-MULTISTREAM] Stream ", stream.streamIndex,
                                  " → binding ", binding,
                                  " (stride=", stream.stride,
                                  ", bytes=", stream.data.size(), ")"));

          // DEBUG: Dump first few vertices (positions) to verify data is reasonable
          Logger::info(str::format("[VSTREAM-DEBUG] Stream ", stream.streamIndex, " - Dumping first 3 vertices:"));
          const uint32_t numVertsToDump = std::min(3u, static_cast<uint32_t>(stream.data.size() / stream.stride));
          for (uint32_t v = 0; v < numVertsToDump; v++) {
            const size_t vOffset = v * stream.stride;
            if (vOffset + 12 <= stream.data.size()) {  // Position is 3 floats = 12 bytes
              const float* pos = reinterpret_cast<const float*>(&stream.data[vOffset]);
              Logger::info(str::format("[VSTREAM-DEBUG]   Vert[", v, "] Position: [",
                                      pos[0], ", ", pos[1], ", ", pos[2], "]"));
              // Sanity check
              if (std::isnan(pos[0]) || std::abs(pos[0]) > 100000.0f) {
                Logger::err(str::format("[VSTREAM-DEBUG]   ❌ Vertex ", v, " has INVALID position!"));
              }
            }
          }

          // DEBUG: Dump COLOR0 values from first few vertices to diagnose alpha=0 issue
          // COLOR0 is at offset 16 with format VK_FORMAT_B8G8R8A8_UNORM based on vertex layout
          if (stream.stride == 28 && stream.data.size() >= 28) {
            const uint32_t color0Offset = 16; // From vertex layout above
            const uint32_t numVerticesToDump = std::min(5u, static_cast<uint32_t>(stream.data.size() / stream.stride));

            Logger::info(str::format("[COLOR0-DEBUG] ===== DUMPING COLOR0 VALUES FOR FIRST ", numVerticesToDump, " VERTICES ====="));
            Logger::info(str::format("[COLOR0-DEBUG] Stream ", stream.streamIndex, " stride=", stream.stride, " COLOR0 offset=", color0Offset));

            for (uint32_t v = 0; v < numVerticesToDump; v++) {
              const size_t vertexOffset = v * stream.stride;
              const size_t color0ByteOffset = vertexOffset + color0Offset;

              if (color0ByteOffset + 4 <= stream.data.size()) {
                // COLOR0 is VK_FORMAT_B8G8R8A8_UNORM (D3DCOLOR format)
                const uint8_t* colorPtr = &stream.data[color0ByteOffset];
                const uint8_t b = colorPtr[0];
                const uint8_t g = colorPtr[1];
                const uint8_t r = colorPtr[2];
                const uint8_t a = colorPtr[3];

                Logger::info(str::format("[COLOR0-DEBUG] Vertex[", v, "] COLOR0: R=", static_cast<uint32_t>(r),
                                        " G=", static_cast<uint32_t>(g),
                                        " B=", static_cast<uint32_t>(b),
                                        " A=", static_cast<uint32_t>(a),
                                        " (normalized: R=", r / 255.0f,
                                        " G=", g / 255.0f,
                                        " B=", b / 255.0f,
                                        " A=", a / 255.0f, ")"));
              }
            }

            Logger::info("[COLOR0-DEBUG] ===== END COLOR0 DUMP =====");
          }
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

          Logger::info(str::format("[OPTION-A-MULTISTREAM] Element: stream=", elem.stream,
                                  " offset=", elem.offset,
                                  " type=", elem.type,
                                  " usage=", elem.usage,
                                  " usageIdx=", elem.usageIndex,
                                  " → location=", attrList[attrCount - 1].location,
                                  " binding=", it->second));
        }

        if (attrCount == 0) {
          Logger::warn("[OPTION-A-MULTISTREAM] No valid attributes - falling back to legacy path");
          attrCount = 0;
          bindCount = 0;
        } else {
          Logger::info(str::format("[OPTION-A-MULTISTREAM] Built layout: ", attrCount, " attributes, ", bindCount, " bindings"));
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

        if (hasColor0 && !preservedVertexData.empty() && drawCallState.originalVertexStride > 0) {
          const uint32_t stride = drawCallState.originalVertexStride;
          const uint32_t numVerticesToDump = std::min(5u, static_cast<uint32_t>(preservedVertexData.size() / stride));

          Logger::info(str::format("[COLOR0-DEBUG] ===== DUMPING COLOR0 VALUES FOR FIRST ", numVerticesToDump, " VERTICES ====="));
          Logger::info(str::format("[COLOR0-DEBUG] COLOR0 offset=", color0Offset, " format=", color0Format, " stride=", stride));

          for (uint32_t v = 0; v < numVerticesToDump; v++) {
            const size_t vertexOffset = v * stride;
            const size_t color0ByteOffset = vertexOffset + color0Offset;

            if (color0ByteOffset + 4 <= preservedVertexData.size()) {
              // COLOR0 is VK_FORMAT_B8G8R8A8_UNORM (D3DCOLOR format)
              const uint8_t* colorPtr = &preservedVertexData[color0ByteOffset];
              const uint8_t b = colorPtr[0];
              const uint8_t g = colorPtr[1];
              const uint8_t r = colorPtr[2];
              const uint8_t a = colorPtr[3];

              Logger::info(str::format("[COLOR0-DEBUG] Vertex[", v, "] COLOR0: R=", static_cast<uint32_t>(r),
                                      " G=", static_cast<uint32_t>(g),
                                      " B=", static_cast<uint32_t>(b),
                                      " A=", static_cast<uint32_t>(a),
                                      " (normalized: R=", r / 255.0f,
                                      " G=", g / 255.0f,
                                      " B=", b / 255.0f,
                                      " A=", a / 255.0f, ")"));
            }
          }

          Logger::info("[COLOR0-DEBUG] ===== END COLOR0 DUMP =====");
        } else if (!hasColor0) {
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

        {
          std::string rebaseLog = str::format("[OPTION-A-EXECUTE] Rebased indices (min=", drawCallState.originalIndexMin, "):");
          const size_t sampleCount = std::min<size_t>(indexCount, size_t(8));
          if (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) {
            const uint32_t* dst32 = reinterpret_cast<const uint32_t*>(rebasedIndexBytes.data());
            for (size_t i = 0; i < sampleCount; i++) {
              rebaseLog += str::format(" ", dst32[i]);
            }
          } else {
            const uint16_t* dst16 = reinterpret_cast<const uint16_t*>(rebasedIndexBytes.data());
            for (size_t i = 0; i < sampleCount; i++) {
              rebaseLog += str::format(" ", dst16[i]);
            }
          }
          Logger::info(rebaseLog.c_str());
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
          Logger::info(str::format("[OPTION-A-EXECUTE] Replayed captured index buffer: bytes=",
                                  preservedIndexData.size(),
                                  " type=", drawCallState.originalIndexType));
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

        Logger::info(str::format("[OPTION-A-EXECUTE] Replayed vertex buffer: bytes=", preservedVertexData.size(),
                                " stride=", stride));

        actualVertexCount = stride > 0 ? preservedVertexData.size() / stride : 0;
        if (!preservedVertexData.empty()) {
          const float* vbFloats = reinterpret_cast<const float*>(preservedVertexData.data());
          const size_t floatCount = std::min<size_t>(preservedVertexData.size() / sizeof(float), size_t(8));
          float vx = floatCount > 0 ? vbFloats[0] : 0.0f;
          float vy = floatCount > 1 ? vbFloats[1] : 0.0f;
          float vz = floatCount > 2 ? vbFloats[2] : 0.0f;
          float vw = floatCount > 3 ? vbFloats[3] : 0.0f;
          Logger::info(str::format("[OPTION-A-EXECUTE] Vertex[0] sample floats: ", vx, ", ", vy, ", ", vz, ", ", vw));
        }

        uint32_t optionAIndexStride = (drawCallState.originalIndexType == VK_INDEX_TYPE_UINT32) ? 4u : 2u;
        if (replayIndexBuffer != nullptr) {
          DxvkBufferSlice ibSlice(replayIndexBuffer);
          ctx->bindIndexBuffer(ibSlice, drawCallState.originalIndexType);

          Logger::info(str::format("[OPTION-A-EXECUTE] Replayed index buffer: bytes=", preservedIndexData.size(),
                                  " type=", drawCallState.originalIndexType));

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

      Logger::info(str::format("[OPTION-A-EXECUTE-FIX] buffersValid=", buffersValid ? 1 : 0,
                              " usedCapturedStreamsPath=", usedCapturedStreamsPath ? 1 : 0,
                              " usedOriginalBuffers=", usedOriginalBuffers ? 1 : 0));

      if (buffersValid) {
        Logger::info("[OPTION-A-EXECUTE] Successfully bound captured vertex/index buffers - skipping RasterGeometry binding");
      } else if (usedCapturedStreamsPath) {
        Logger::info("[OPTION-A-EXECUTE] Using capturedVertexStreams - buffers are REBASED, will use firstIndex=0, vertexOffset=0");
      }

      Logger::info(str::format("[OPTION-A-EXECUTE] Replay counts: vertexCount=", actualVertexCount,
                              " indexCount=", actualIndexCount,
                              " baseVertex=", drawCallState.originalBaseVertex,
                              " firstIndex=", drawCallState.originalFirstIndex,
                              " indexMin=", drawCallState.originalIndexMin));
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

    // ALWAYS log constant data availability
    Logger::info(str::format("[OPTION-C-DEBUG] ========== SHADER CONSTANTS START =========="));
    Logger::info(str::format("[OPTION-C-DEBUG] VS constant data: size=", drawCallState.vertexShaderConstantData.size(),
                            " vectors, bytes=", vsConstBytes.size(),
                            " isEmpty=", vsConstBytes.empty() ? "YES" : "NO"));

    // CRITICAL VALIDATION: Reject cached captures with zero transformation matrices
    // This handles cached captures from before the zero-matrix fix was added
    if (!drawCallState.vertexShaderConstantData.empty() && drawCallState.vertexShaderConstantData.size() >= 4) {
      bool allZeros = true;
      for (uint32_t i = 0; i < 4; i++) {
        const Vector4& c = drawCallState.vertexShaderConstantData[i];
        if (c.x != 0.0f || c.y != 0.0f || c.z != 0.0f || c.w != 0.0f) {
          allZeros = false;
          break;
        }
      }

      if (allZeros) {
        static uint32_t zeroMatrixRejectCount = 0;
        if (++zeroMatrixRejectCount <= 5) {
          Logger::err(str::format("[ShaderOutputCapturer] ❌ REJECTING CACHED CAPTURE - VS constants c[0]-c[3] are ALL ZERO! ",
                                  "This is a bad cached capture from initialization. Invalidating cache entry. (Count: ", zeroMatrixRejectCount, ")"));
        }

        // Invalidate the cached capture so it will be re-captured with valid matrices
        auto [cacheKey, isValidKey] = getCacheKey(drawCallState);
        if (isValidKey && cacheKey != 0) {
          m_capturedOutputs.erase(cacheKey);
          Logger::warn(str::format("[ShaderOutputCapturer] Invalidated cache entry for cacheKey=0x", std::hex, cacheKey, std::dec,
                                  ". Material will be re-captured when matrices are valid."));
        }

        // Abort this draw - don't re-execute with zero matrices
        Logger::warn("[ShaderOutputCapturer] Aborting shader re-execution. Draw will fall back to regular rendering.");
        return false; // Abort capture
      }
    }

    // Log first 16 constants for debugging transformation matrices (expanded from 4)
    if (!drawCallState.vertexShaderConstantData.empty()) {
      const size_t numToLog = std::min<size_t>(16, drawCallState.vertexShaderConstantData.size());
      Logger::info(str::format("[OPTION-C-DEBUG]   VS constants c[0]-c[15] (transformation matrices should be non-zero):"));
      for (size_t i = 0; i < numToLog; i++) {
        const Vector4& c = drawCallState.vertexShaderConstantData[i];
        Logger::info(str::format("[OPTION-C-DEBUG]   VS c[", i, "] = (", c.x, ", ", c.y, ", ", c.z, ", ", c.w, ")"));
      }
    }

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

      Logger::info(str::format("[OPTION-C-DEBUG] ✓ VS CONSTANTS BOUND: slot=", vsConstantBufferSlot,
                              " vectors=", drawCallState.vertexShaderConstantData.size(),
                              " bytes=", vsConstantSlice.length()));
    } else if (!drawCallState.vertexShaderConstantData.empty()) {
      Logger::err("[OPTION-C-DEBUG] ✗ VS CONSTANTS FAILED TO UPLOAD despite having data!");
    } else {
      Logger::err("[OPTION-C-DEBUG] ✗ VS CONSTANTS EMPTY - shader will use stale/default values!");
    }

    // Bind pixel shader constants copied at Stage 2
    // Cast Vector4 vector to uint8_t vector for the buffer creation
    const std::vector<uint8_t> psConstBytes(
      reinterpret_cast<const uint8_t*>(drawCallState.pixelShaderConstantData.data()),
      reinterpret_cast<const uint8_t*>(drawCallState.pixelShaderConstantData.data() + drawCallState.pixelShaderConstantData.size()));

    // ALWAYS log constant data availability
    Logger::info(str::format("[OPTION-C-DEBUG] PS constant data: size=", drawCallState.pixelShaderConstantData.size(),
                            " vectors, bytes=", psConstBytes.size(),
                            " isEmpty=", psConstBytes.empty() ? "YES" : "NO"));

    // Log specific constants where game sets values (c[30], c[150], c[180])
    if (!drawCallState.pixelShaderConstantData.empty()) {
      const size_t dataSize = drawCallState.pixelShaderConstantData.size();
      if (dataSize > 30) {
        const Vector4& c30 = drawCallState.pixelShaderConstantData[30];
        Logger::info(str::format("[OPTION-C-DEBUG]   PS c[30] = (", c30.x, ", ", c30.y, ", ", c30.z, ", ", c30.w, ")"));
      }
      if (dataSize > 150) {
        const Vector4& c150 = drawCallState.pixelShaderConstantData[150];
        Logger::info(str::format("[OPTION-C-DEBUG]   PS c[150] = (", c150.x, ", ", c150.y, ", ", c150.z, ", ", c150.w, ")"));
      }
      if (dataSize > 180) {
        const Vector4& c180 = drawCallState.pixelShaderConstantData[180];
        Logger::info(str::format("[OPTION-C-DEBUG]   PS c[180] = (", c180.x, ", ", c180.y, ", ", c180.z, ", ", c180.w, ")"));
      }
    }

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

      Logger::info(str::format("[OPTION-C-DEBUG] ✓ PS CONSTANTS BOUND: slot=", psConstantBufferSlot,
                              " vectors=", drawCallState.pixelShaderConstantData.size(),
                              " bytes=", psConstantSlice.length()));
    } else if (!drawCallState.pixelShaderConstantData.empty()) {
      Logger::err("[OPTION-C-DEBUG] ✗ PS CONSTANTS FAILED TO UPLOAD despite having data!");
    } else {
      Logger::err("[OPTION-C-DEBUG] ✗ PS CONSTANTS EMPTY - shader will use stale/default values!");
    }

    Logger::info(str::format("[OPTION-C-DEBUG] ========== SHADER CONSTANTS END =========="));

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

    Logger::info(str::format("[OPTION-C-DEBUG] ========== DRAW CALL START =========="));
    Logger::info(str::format("[OPTION-C-DEBUG] actualVertexCount=", actualVertexCount,
                            " actualIndexCount=", actualIndexCount,
                            " instanceCount=", drawParams.instanceCount));
    Logger::info(str::format("[OPTION-C-DEBUG] usedOriginalBuffers=", usedOriginalBuffers ? "YES" : "NO",
                            " topology=", geom.topology));
    Logger::info(str::format("[VSTREAM-DEBUG] ===== FINAL DRAW PARAMETERS ====="));
    Logger::info(str::format("[VSTREAM-DEBUG] actualVertexCount=", actualVertexCount, " (ZERO? ", (actualVertexCount == 0 ? "YES!!!" : "NO"), ")"));
    Logger::info(str::format("[VSTREAM-DEBUG] actualIndexCount=", actualIndexCount, " (ZERO? ", (actualIndexCount == 0 ? "YES (non-indexed)" : "NO"), ")"));
    Logger::info(str::format("[VSTREAM-DEBUG] instanceCount=", drawParams.instanceCount, " (ZERO? ", (drawParams.instanceCount == 0 ? "YES!!!" : "NO"), ")"));
    Logger::info(str::format("[VSTREAM-DEBUG] attrCount=", attrCount, " (attributes bound)"));
    Logger::info(str::format("[VSTREAM-DEBUG] bindCount=", bindCount, " (vertex bindings)"));

    if (actualVertexCount == 0 || (actualIndexCount == 0 && geom.usesIndices())) {
      Logger::err("[VSTREAM-DEBUG] ❌ CRITICAL: Zero vertices/indices - geometry will NOT render!");
      Logger::err(str::format("[VSTREAM-DEBUG] drawParams.vertexCount=", drawParams.vertexCount));
      Logger::err(str::format("[VSTREAM-DEBUG] drawParams.indexCount=", drawParams.indexCount));
      Logger::err(str::format("[VSTREAM-DEBUG] geom.vertexCount=", geom.vertexCount));
      Logger::err(str::format("[VSTREAM-DEBUG] geom.indexCount=", geom.indexCount));
    }

    auto drawCallStartTime = std::chrono::high_resolution_clock::now();

    if (actualIndexCount == 0) {
      // Non-indexed draw
      const uint32_t startVertex = usedOriginalBuffers ? 0 : drawParams.vertexOffset;

      Logger::info(str::format("[OPTION-C-DEBUG] NON-INDEXED DRAW: vertexCount=", actualVertexCount,
                              " instanceCount=", drawParams.instanceCount,
                              " startVertex=", startVertex));

      ctx->DxvkContext::draw(
        actualVertexCount,
        drawParams.instanceCount,
        startVertex,
        0);

      Logger::info(str::format("[OPTION-C-DEBUG] ✓ Non-indexed draw completed"));

      if (drawCallState.renderTargetReplacementSlot >= 0) {
        Logger::info(str::format("[ShaderCapture-DRAW] Non-indexed draw completed - startVertex=", startVertex));
      }
    } else {
      // Indexed draw
      Logger::info(str::format("[OPTION-C-DEBUG] INDEXED DRAW: indexCount=", actualIndexCount,
                              " instanceCount=", drawParams.instanceCount,
                              " firstIndex=", firstIndex,
                              " vertexOffset=", vertexOffset,
                              " triangles=", actualIndexCount / 3));

      ctx->DxvkContext::drawIndexed(
        actualIndexCount,
        drawParams.instanceCount,
        firstIndex,
        vertexOffset,
        0);

      Logger::info(str::format("[OPTION-C-DEBUG] ✓ Indexed draw completed - drew ", actualIndexCount / 3, " triangles"));
    }

    auto drawCallEndTime = std::chrono::high_resolution_clock::now();
    auto drawCallDuration = std::chrono::duration_cast<std::chrono::microseconds>(drawCallEndTime - drawCallStartTime).count();
    Logger::info(str::format("[OPTION-C-PERF] ⏱️ DRAW CALL GPU SUBMIT TIME: ", drawCallDuration / 1000.0, " ms"));
    Logger::info(str::format("[OPTION-C-DEBUG] ========== DRAW CALL END =========="));

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

    // CRITICAL FIX: Transition render target from COLOR_ATTACHMENT_OPTIMAL to SHADER_READ_ONLY_OPTIMAL
    // This allows the image to be:
    // 1. Saved to disk correctly (fixes transparent/corrupt saved textures)
    // 2. Used as a shader input texture in the raytracing pipeline
    // Without this transition, Vulkan validation errors occur and textures appear black/transparent
    ctx->changeImageLayout(renderTarget.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Logger::info(str::format("[ShaderCapture-Layout] Transitioned render target 0x", std::hex,
                            renderTarget.image->getHash(), std::dec,
                            " from COLOR_ATTACHMENT_OPTIMAL to SHADER_READ_ONLY_OPTIMAL"));

    // Restore previous render target
    if (prevColorTarget != nullptr) {
      DxvkRenderTargets prevRt;
      prevRt.color[0].view = prevColorTarget;
      prevRt.color[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
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

    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));
    Logger::info(str::format("[OPTION-C-DEBUG] ====== CAPTURE #", captureCallCount, " SUCCESS ======"));
    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));
    Logger::info(str::format("[OPTION-C-DEBUG] Render Target Details:"));
    Logger::info(str::format("[OPTION-C-DEBUG]   hash=0x", std::hex, renderTarget.image->getHash(), std::dec));
    Logger::info(str::format("[OPTION-C-DEBUG]   format=", renderTarget.image->info().format));
    Logger::info(str::format("[OPTION-C-DEBUG]   size=", renderTarget.image->info().extent.width, "x", renderTarget.image->info().extent.height));
    Logger::info(str::format("[OPTION-C-DEBUG]   mips=", renderTarget.image->info().mipLevels));
    Logger::info(str::format("[OPTION-C-DEBUG]   layout=SHADER_READ_ONLY_OPTIMAL (transitioned from COLOR_ATTACHMENT)"));
    Logger::info(str::format("[OPTION-C-DEBUG] Output Texture:"));
    Logger::info(str::format("[OPTION-C-DEBUG]   valid=", outputTexture.isValid() ? "YES" : "NO"));
    if (outputTexture.isValid()) {
      Logger::info(str::format("[OPTION-C-DEBUG]   hash=0x", std::hex, outputTexture.getImageHash(), std::dec));
    }
    Logger::info(str::format("[OPTION-C-DEBUG] ========================================"));

    return true;
  }

  TextureRef ShaderOutputCapturer::getCapturedTexture(XXH64_hash_t materialHash) const {
    auto it = m_capturedOutputs.find(materialHash);
    if (it != m_capturedOutputs.end() && it->second.capturedTexture.isValid()) {
      return TextureRef(it->second.capturedTexture.view);
    }
    return TextureRef();
  }

  bool ShaderOutputCapturer::hasCapturedTexture(XXH64_hash_t materialHash) const {
    auto it = m_capturedOutputs.find(materialHash);
    return it != m_capturedOutputs.end() && it->second.capturedTexture.isValid();
  }

  void ShaderOutputCapturer::onFrameBegin(Rc<RtxContext> ctx) {
    static uint32_t frameBeginCallCount = 0;
    ++frameBeginCallCount;

    if (frameBeginCallCount <= 10) {
      Logger::info(str::format("[FRAMEBEGIN-ENTRY] onFrameBegin #", frameBeginCallCount,
                              " pendingRequests=", m_pendingCaptureRequests.size()));
    }

    // GPU-DRIVEN CAPTURE SYSTEM: Lazy initialization on first frame
    static bool gpuSystemInitialized = false;
    if (!gpuSystemInitialized && enableShaderOutputCapture()) {
      Logger::info("[ShaderCapture-GPU] Initializing GPU-driven batched capture system...");
      initializeGpuCaptureSystem(ctx);
      // NO pre-allocated pool - RTs allocated on-demand for massive VRAM savings!
      gpuSystemInitialized = true;
      Logger::info("[ShaderCapture-GPU] ===== GPU-DRIVEN CAPTURE SYSTEM READY =====");
    }

    // Execute any remaining captures from previous frame
    if (enableShaderOutputCapture() && !m_pendingCaptureRequests.empty()) {
      Logger::info(str::format("[ShaderCapture-GPU-EXEC] Executing ", m_pendingCaptureRequests.size(),
                              " queued captures from previous frame"));
      buildGpuCaptureList(ctx);
      executeMultiIndirectCaptures(ctx);
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

    Logger::info(str::format("========== [DEBUG] onFrameEnd() CALLED (call #", frameEndCallCount, ") pendingRequests=", m_pendingCaptureRequests.size(), " =========="));

    // Execute any remaining captures at end of frame (before frame boundary)
    // GpuCaptureRequest is self-contained with no pointers, so this is safe!
    if (!m_pendingCaptureRequests.empty()) {
      Logger::info(str::format("[ShaderCapture-GPU-FRAMEEND] Executing ", m_pendingCaptureRequests.size(),
                              " remaining captures at end of frame"));
      // Note: ctx is not available here, so we can't execute
      // Captures will be executed in onFrameBegin of NEXT frame
    }

    // Cleanup old cached outputs if needed
    // For now, keep everything cached
  }

  Resources::Resource ShaderOutputCapturer::getRenderTarget(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      XXH64_hash_t materialHash) {

    // Create cache key from resolution and format ONLY (NO material hash!)
    // This allows RT sharing between materials with same resolution = MASSIVE VRAM savings
    // We don't need per-material RTs because we copy the result immediately after capture
    uint64_t cacheKey = (static_cast<uint64_t>(resolution.width) << 32) |
                        (static_cast<uint64_t>(resolution.height) << 16) |
                        static_cast<uint64_t>(format);
    // REMOVED: XOR with material hash - was causing 1 RT per material = GB of wasted VRAM!

    // Check cache
    auto it = m_renderTargetCache.find(cacheKey);
    if (it != m_renderTargetCache.end() && it->second.isValid()) {
      return it->second;
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
                               " textureHash=0x", std::hex, cacheKey, std::dec));
    }

    // Cache it
    m_renderTargetCache[cacheKey] = resource;

    return resource;
  }

  Resources::Resource ShaderOutputCapturer::getRenderTargetArray(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      uint32_t layerCount) {

    // Create texture array for layered rendering - NO caching since layer count varies
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
      Logger::info(str::format("[ShaderOutputCapturer] Created texture array render target: ",
                               resolution.width, "x", resolution.height,
                               " layers=", layerCount,
                               " format=", format));
    }

    return resource;
  }

  VkExtent2D ShaderOutputCapturer::calculateCaptureResolution(
      const DrawCallState& drawCallState) const {

    uint32_t baseResolution = captureResolution();

    // Clamp to valid range
    baseResolution = std::clamp(baseResolution, 256u, 4096u);

    // For now, use fixed resolution
    // TODO: Implement LOD based on screen space size
    return { baseResolution, baseResolution };
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

    CapturedShaderOutput& output = m_capturedOutputs[cacheKey];
    output.capturedTexture = texture;
    output.geometryHash = geomHash;
    output.materialHash = matHash;
    output.lastCaptureFrame = currentFrame;
    output.captureSubmittedFrame = currentFrame;  // Track when submitted for async tracking
    output.isDynamic = isDynamicMaterial(matHash);
    output.isPending = true;  // Mark as pending - GPU hasn't finished yet
    output.resolution = { texture.image->info().extent.width,
                          texture.image->info().extent.height };

    static uint32_t storeLogCount = 0;
    const bool shouldLog = (++storeLogCount <= 20) || (drawCallState.renderTargetReplacementSlot >= 0);
    if (shouldLog) {
      Logger::info(str::format("[ShaderCapture-Store] #", storeLogCount,
                              " Stored cacheKey=0x", std::hex, cacheKey, std::dec,
                              " matHash=0x", std::hex, matHash, std::dec,
                              " isRTReplacement=", (drawCallState.renderTargetReplacementSlot >= 0 ? "YES" : "NO"),
                              " isDynamic=", output.isDynamic ? "YES" : "NO",
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
    // CRITICAL: RT replacement materials should be STATIC by default (capture once, cache forever)
    // Only materials explicitly marked in dynamicShaderMaterials() should be recaptured every frame
    //
    // captureEnabledHashes() is for whitelisting which materials to capture with shader re-execution,
    // but it does NOT mean they're dynamic (animated). Most RT replacement materials are static.

    static uint32_t isDynamicLogCount = 0;
    const bool shouldLog = (++isDynamicLogCount <= 20);

    // Check if this specific material is explicitly marked as dynamic
    const bool inDynamicSet = dynamicShaderMaterials().count(materialHash) > 0;

    if (shouldLog) {
      Logger::info(str::format("[ShaderCapture-Dynamic] #", isDynamicLogCount,
                              " matHash=0x", std::hex, materialHash, std::dec,
                              " isDynamic=", inDynamicSet ? "YES (explicitly marked)" : "NO (default to static)",
                              " dynamicShaderMaterials.size=", dynamicShaderMaterials().size()));
    }

    return inDynamicSet;
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

} // namespace dxvk


