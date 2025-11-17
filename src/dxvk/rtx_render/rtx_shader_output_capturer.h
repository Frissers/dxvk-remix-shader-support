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
#pragma once

#include "rtx_resources.h"
#include "rtx_option.h"
#include "../util/rc/util_rc_ptr.h"
#include "../../util/util_matrix.h"

namespace dxvk {
  class RtxContext;
  struct DrawCallState;
  struct DrawParameters;
  struct DxvkRaytracingInstanceState;

  /**
   * \brief Shader Output Capturer
   *
   * Captures pixel shader output from game draw calls and stores it as textures
   * that can be used as albedo during path tracing. This allows shader effects
   * (water, animated textures, wet surfaces, etc.) to work with RTX Remix.
   */
  class ShaderOutputCapturer {
  public:
    ShaderOutputCapturer();
    ~ShaderOutputCapturer();

    // Check if we should capture shader output for this draw call (Stage 2 - no cache check)
    static bool shouldCaptureStatic(const DrawCallState& drawCallState);

    // Check if we should capture shader output for this draw call (Stage 4 - with cache check)
    bool shouldCapture(const DrawCallState& drawCallState) const;

    // Capture shader output for a draw call and return the captured texture
    bool captureDrawCall(
      Rc<RtxContext> ctx,
      const DxvkRaytracingInstanceState& rtState,
      const DrawCallState& drawCallState,
      const DrawParameters& drawParams,
      TextureRef& outputTexture);

    // Get previously captured texture for a material
    TextureRef getCapturedTexture(XXH64_hash_t materialHash) const;

    // Check if this material has been captured
    bool hasCapturedTexture(XXH64_hash_t materialHash) const;

    // Check if we need to re-capture this draw call (for animated shaders)
    bool needsRecapture(const DrawCallState& drawCallState, uint32_t currentFrame) const;

    // PROPER TEXTURE REPLACEMENT API
    // Register a texture replacement: when originalTextureHash is referenced, use replacementTexture instead
    void registerTextureReplacement(XXH64_hash_t originalTextureHash, const TextureRef& replacementTexture);

    // Get replacement texture for a given original texture hash
    // Returns invalid TextureRef if no replacement exists
    TextureRef getReplacementTexture(XXH64_hash_t originalTextureHash) const;

    // Check if a texture has a replacement
    bool hasReplacementTexture(XXH64_hash_t originalTextureHash) const;

    // Called at start of frame
    void onFrameBegin(Rc<RtxContext> ctx);

    // Called at end of frame
    void onFrameEnd();

    // ImGui settings
    void showImguiSettings();

    // RTX Options
    RTX_OPTION("rtx.shaderCapture", bool, enableShaderOutputCapture, true,
      "Enables capturing of pixel shader output to use as albedo textures in path tracing.\n"
      "This allows shader effects (water, animated textures, etc.) to work with RTX.\n"
      "Uses proper texture replacement system to avoid braille artifacts.\n"
      "Enabled by default with automatic VRAM cleanup every 60 frames.");

    RTX_OPTION("rtx.shaderCapture", uint32_t, captureResolution, 1024,
      "Resolution of captured shader output textures. Higher = better quality but more VRAM.\n"
      "Recommended: 512-2048. Range: 256-4096");

    RTX_OPTION("rtx.shaderCapture", bool, dynamicCaptureOnly, false,
      "Only capture shader output for materials marked as dynamic.\n"
      "Reduces overhead for static materials.\n"
      "TEMP: Disabled to force re-capture with new pipeline state.");

    RTX_OPTION("rtx.shaderCapture", uint32_t, maxCapturesPerFrame, 1000,
      "Maximum number of shader captures per frame to prevent performance spikes.\n"
      "Set high (1000) for many dynamic materials. TODO: Async execution to spread cost.");

    RTX_OPTION("rtx.shaderCapture", uint32_t, recaptureInterval, 1,
      "Number of frames between re-captures for dynamic materials.\n"
      "1 = every frame (smooth animation), higher = better performance.");

    RTX_OPTION("rtx.shaderCapture", bool, captureAllDraws, true,
      "Capture shader output for all draw calls (except UI).\n"
      "Enabled by default to ensure shader effects work properly.\n"
      "UI draws (pixel shader only, no vertex shader) are automatically excluded.");

    RTX_OPTION_ENV("rtx.shaderCapture", fast_unordered_set, captureEnabledHashes, {},
      "RTX_SHADER_CAPTURE_ENABLED_HASHES",
      "List of material hashes to capture shader output for.\n"
      "Add material hash from Remix UI. Use 0xALL to capture all (very expensive).");

    RTX_OPTION_ENV("rtx.shaderCapture", fast_unordered_set, dynamicShaderMaterials, {},
      "RTX_DYNAMIC_SHADER_MATERIALS",
      "List of material hashes that have animated shaders (water, etc.).\n"
      "These will be re-captured periodically based on recaptureInterval.");

  private:
    // Helper to get cache key for a draw call
    // For RT replacements, use source texture hash; for RT feedback, use originalRT hash; otherwise use material hash
    // Returns a pair: (cacheKey, isValid)
    // isValid=false means the RT replacement texture is invalid and should be rejected
    std::pair<XXH64_hash_t, bool> getCacheKey(const DrawCallState& drawCallState) const {
      if (drawCallState.renderTargetReplacementSlot >= 0) {
        const TextureRef& replacementTexture = drawCallState.getMaterialData().getColorTexture();
        if (replacementTexture.isValid()) {
          return {replacementTexture.getImageHash(), true};
        }
        return {0, false}; // Invalid RT replacement
      }
      // Check for render target feedback case: no replacement found but slot 0 was a render target
      // Use the originalRT hash as cache key to differentiate draws to different RTs
      if (drawCallState.originalRenderTargetHash != 0) {
        return {drawCallState.originalRenderTargetHash, true};
      }
      // For regular materials, hash can legitimately be 0, so it's always valid
      return {drawCallState.getMaterialData().getHash(), true};
    }

    struct CapturedShaderOutput {
      Resources::Resource capturedTexture;  // Albedo output (always captured)
      Resources::Resource capturedNormals;  // Normal map output (if material has normals)
      Resources::Resource capturedRoughness; // Roughness output (if material has roughness)
      Resources::Resource capturedEmissive;  // Emissive output (if material has emissive)
      XXH64_hash_t geometryHash;
      XXH64_hash_t materialHash;
      uint32_t lastCaptureFrame;
      uint32_t captureSubmittedFrame;  // Frame when capture was submitted to GPU
      bool isDynamic;
      bool isPending;                   // True if GPU hasn't finished capture yet
      VkExtent2D resolution;
      uint32_t arrayLayer = 0;          // Layer index if capturedTexture is an array (0 if individual)
      bool isArrayLayer = false;        // True if this references a layer in a texture array

      // Material complexity flags for efficient detection
      bool hasNormals = false;
      bool hasRoughness = false;
      bool hasEmissive = false;
    };

    // Storage for captured outputs (mutable to allow updating isPending flag from const functions)
    mutable fast_unordered_cache<CapturedShaderOutput> m_capturedOutputs;

    // PROPER TEXTURE REPLACEMENT: Map from original RT texture hash to captured replacement texture
    // When the game references originalTextureHash, we return replacementTexture instead
    std::unordered_map<XXH64_hash_t, TextureRef> m_textureReplacements;

    // ======== GPU-DRIVEN MULTI-INDIRECT CAPTURE SYSTEM (MegaGeometry-style) ========

    // GPU Capture Request - SELF-CONTAINED (no pointers, all data copied)
    // This struct owns all the data needed for capture, avoiding dangling pointer issues
    struct GpuCaptureRequest {
      XXH64_hash_t materialHash;      // Material to capture
      XXH64_hash_t geometryHash;      // Geometry hash
      XXH64_hash_t textureHash;       // Primary texture hash
      uint32_t drawCallIndex;         // Index into draw call data
      uint32_t renderTargetIndex;     // Index into RT pool
      uint32_t vertexOffset;          // Vertex buffer offset
      uint32_t vertexCount;           // Number of vertices
      uint32_t indexOffset;           // Index buffer offset  (if indexed)
      uint32_t indexCount;            // Number of indices (if indexed)
      VkExtent2D resolution;          // Capture resolution
      uint32_t flags;                 // Capture flags (indexed, dynamic, etc.)
      bool isDynamic = false;         // Is this a dynamic material?

      // GEOMETRY DATA (Rc-counted, safe to copy)
      DxvkBufferSlice vertexBuffer;   // Position buffer
      DxvkBufferSlice indexBuffer;    // Index buffer (if indexed)
      DxvkBufferSlice texcoordBuffer; // UV buffer
      DxvkBufferSlice normalBuffer;   // Normal buffer
      uint32_t vertexStride = 0;      // Vertex buffer stride
      uint32_t texcoordStride = 0;    // Texcoord buffer stride
      VkIndexType indexType = VK_INDEX_TYPE_NONE_KHR;  // Index type

      // TEXTURE DATA (Rc-counted, safe to copy)
      TextureRef colorTexture;        // Primary color texture

      // VIEWPORT/SCISSOR STATE
      VkViewport viewport;
      VkRect2D scissor;

      // REPLACEMENT BUFFER SUPPORT (optional)
      DxvkBufferSlice replacementVertexBuffer;
      DxvkBufferSlice replacementTexcoordBuffer;
      DxvkBufferSlice replacementNormalBuffer;
      DxvkBufferSlice replacementIndexBuffer;
      uint32_t replacementVertexStride = 0;
      uint32_t replacementTexcoordStride = 0;
      VkIndexType replacementIndexType = VK_INDEX_TYPE_NONE_KHR;
    };

    // Indirect Draw Args - filled by GPU compute shader
    struct IndirectDrawArgs {
      uint32_t vertexCount;
      uint32_t instanceCount;
      uint32_t firstVertex;
      uint32_t firstInstance;
    };

    // Indirect Indexed Draw Args - for indexed draws
    struct IndirectIndexedDrawArgs {
      uint32_t indexCount;
      uint32_t instanceCount;
      uint32_t firstIndex;
      int32_t  vertexOffset;
      uint32_t firstInstance;
    };

    // GPU Counters - atomic counters for GPU work tracking
    struct GpuCaptureCounters {
      uint32_t totalCaptureRequests;    // Total requests queued
      uint32_t processedCaptures;       // Captures executed
      uint32_t skippedCaptures;         // Captures skipped (already cached)
      uint32_t failedCaptures;          // Captures that failed
      uint32_t padding[12];             // Align to cache line
    };

    // DELETED: Pre-allocated pool wasted ~8GB VRAM!
    // Now using on-demand allocation via m_renderTargetCache (see below)

    // GPU buffers for multi-indirect dispatch
    DxvkBufferSlice m_captureRequestsBuffer;      // GpuCaptureRequest array
    DxvkBufferSlice m_indirectDrawArgsBuffer;     // IndirectDrawArgs array (GPU fills this)
    DxvkBufferSlice m_indirectIndexedDrawArgsBuffer; // IndirectIndexedDrawArgs array
    DxvkBufferSlice m_captureCountersBuffer;      // GpuCaptureCounters (GPU atomics)

    // OPTIMIZED: Persistent indirect buffers for multi-draw-indirect (reused every frame)
    Rc<DxvkBuffer> m_persistentIndirectBuffer;         // For non-indexed draws
    Rc<DxvkBuffer> m_persistentIndexedIndirectBuffer;  // For indexed draws
    size_t m_persistentIndirectBufferSize = 0;
    size_t m_persistentIndexedIndirectBufferSize = 0;

    // CPU-side request queue (built during frame, uploaded to GPU)
    std::vector<GpuCaptureRequest> m_pendingCaptureRequests;

    // GPU-driven capture pipeline
    void initializeGpuCaptureSystem(Rc<RtxContext> ctx);
    void shutdownGpuCaptureSystem();
    // DELETED: allocateRenderTargetPool, allocateRenderTargetFromPool - using on-demand allocation now
    void buildGpuCaptureList(Rc<RtxContext> ctx);
    void executeMultiIndirectCaptures(Rc<RtxContext> ctx);

    // Helper functions for maximum performance batching
    void setCommonPipelineState(Rc<RtxContext> ctx, const GpuCaptureRequest& request);
    void bindGeometryBuffers(Rc<RtxContext> ctx, const GpuCaptureRequest& request);

    // Per-frame capture counter
    uint32_t m_capturesThisFrame = 0;

    // Frame counter for delaying RT feedback captures
    uint32_t m_currentFrame = 0;

    // Create or get cached render target
    Resources::Resource getRenderTarget(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      XXH64_hash_t materialHash);

    // Create texture array render target for layered rendering
    Resources::Resource getRenderTargetArray(
      Rc<RtxContext> ctx,
      VkExtent2D resolution,
      VkFormat format,
      uint32_t layerCount);

    // Calculate appropriate capture resolution for a draw call
    VkExtent2D calculateCaptureResolution(
      const DrawCallState& drawCallState) const;

    // Calculate projection matrix for UV space rendering
    Matrix4 calculateUVSpaceProjection(
      const DrawCallState& drawCallState) const;

    // Store captured output in cache
    void storeCapturedOutput(
      Rc<RtxContext> ctx,
      const DrawCallState& drawCallState,
      const Resources::Resource& texture,
      uint32_t currentFrame);

    // Check if material is marked as dynamic
    bool isDynamicMaterial(XXH64_hash_t materialHash) const;

    // Cache of render targets by resolution
    std::unordered_map<uint64_t, Resources::Resource> m_renderTargetCache;

    // OPTIMIZATION 1: Cache texture arrays by (resolution, layerCount)
    // Key format: resolution (32 bits) | layerCount (32 bits)
    std::unordered_map<uint64_t, Resources::Resource> m_renderTargetArrayCache;

    // OPTIMIZATION 2: Texture array pool for layer allocation
    struct TextureArrayPool {
      Resources::Resource arrayResource;  // The texture array
      uint32_t totalLayers;               // Total layers in array
      uint64_t usedLayersMask;            // Bitfield of used layers (supports up to 64 layers)
      uint32_t allocatedLayers;           // Count of allocated layers
      VkExtent2D resolution;              // Resolution of this pool
    };
    std::vector<TextureArrayPool> m_arrayPools;  // Pool of pre-allocated arrays
    static constexpr uint32_t POOL_LAYERS_PER_ARRAY = 64;
    static constexpr uint32_t MAX_POOL_ARRAYS = 8;

    // OPTIMIZATION 3: Hash-based capture dirty detection
    std::unordered_map<XXH64_hash_t, XXH64_hash_t> m_lastCaptureContentHash;

    // Helper to allocate layers from pool
    struct LayerAllocation {
      Resources::Resource arrayResource;
      uint32_t startLayer;
      uint32_t layerCount;
      size_t poolIndex;
      bool valid = false;
    };
    LayerAllocation allocateLayersFromPool(Rc<RtxContext> ctx, VkExtent2D resolution, uint32_t layerCount);
    void freeLayersToPool(const LayerAllocation& allocation);

    // Vertex shader for gl_Layer output (instanced multi-draw-indirect)
    Rc<DxvkShader> m_layerRoutingVertexShader;
    void initializeLayerRoutingShader(Rc<DxvkDevice> device);

    // Helper to compute UV bounds from geometry
    void computeUVBounds(
      const RasterGeometry& geom,
      Vector2& uvMin,
      Vector2& uvMax) const;
  };

} // namespace dxvk
