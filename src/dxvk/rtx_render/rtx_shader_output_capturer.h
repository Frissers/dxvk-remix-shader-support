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
      "Enabled by default - disable if you experience issues.");

    RTX_OPTION("rtx.shaderCapture", uint32_t, captureResolution, 1024,
      "Resolution of captured shader output textures. Higher = better quality but more VRAM.\n"
      "Recommended: 512-2048. Range: 256-4096");

    RTX_OPTION("rtx.shaderCapture", bool, dynamicCaptureOnly, false,
      "Only capture shader output for materials marked as dynamic.\n"
      "Reduces overhead for static materials.\n"
      "TEMP: Disabled to force re-capture with new pipeline state.");

    RTX_OPTION("rtx.shaderCapture", uint32_t, maxCapturesPerFrame, 100,
      "Maximum number of shader captures per frame to prevent performance spikes.\n"
      "Remaining captures will be done in subsequent frames.");

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
      Resources::Resource capturedTexture;
      XXH64_hash_t geometryHash;
      XXH64_hash_t materialHash;
      uint32_t lastCaptureFrame;
      bool isDynamic;
      VkExtent2D resolution;
    };

    // Storage for captured outputs
    fast_unordered_cache<CapturedShaderOutput> m_capturedOutputs;

    // PROPER TEXTURE REPLACEMENT: Map from original RT texture hash to captured replacement texture
    // When the game references originalTextureHash, we return replacementTexture instead
    std::unordered_map<XXH64_hash_t, TextureRef> m_textureReplacements;

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

    // Helper to compute UV bounds from geometry
    void computeUVBounds(
      const RasterGeometry& geom,
      Vector2& uvMin,
      Vector2& uvMax) const;
  };

} // namespace dxvk
