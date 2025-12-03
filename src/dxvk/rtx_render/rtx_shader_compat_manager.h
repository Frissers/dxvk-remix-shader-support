/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <unordered_map>
#include <mutex>
#include <atomic>

#include "rtx_geometry_identity.h"
#include "rtx_context.h"
#include "../dxvk_device.h"

namespace dxvk {

  // Forward declarations
  struct DrawCallState;
  class RtxContext;

  /**
   * Configuration for the geometry identity system
   */
  struct GeometryIdentityConfig {
    // Stability detection
    uint32_t stabilityCheckFrames = kStabilityCheckFrames;
    uint32_t maxStableVariations = kMaxStableVariations;

    // Bounds computation
    uint32_t boundsSampleCount = 64;        // Vertices to sample for bounds
    bool computeBoundsEveryFrame = false;   // For moving objects

    // UV capture
    bool enableUVCapture = true;
    bool detectAnimatedUVs = true;

    // Texture handling
    bool enableTextureBaking = true;
    bool useMemoryAliasing = true;          // Reuse original texture memory

    // Culling
    bool overrideRemixCulling = true;
    bool useFrustumCullingOnly = true;

    // Debug
    bool enableDebugLogging = false;
    bool visualizeStableIds = false;
  };

  /**
   * Statistics for monitoring system performance
   */
  struct GeometryIdentityStats {
    std::atomic<uint64_t> totalDrawCallsProcessed{0};
    std::atomic<uint64_t> cacheHits{0};
    std::atomic<uint64_t> cacheMisses{0};
    std::atomic<uint64_t> stableIdComputations{0};
    std::atomic<uint64_t> stabilityChecks{0};
    std::atomic<uint64_t> boundsComputations{0};
    std::atomic<uint64_t> uvCaptures{0};
    std::atomic<uint64_t> textureBakes{0};

    // Memory tracking
    std::atomic<uint64_t> geometryRecordBytes{0};
    std::atomic<uint64_t> capturedUVBytes{0};
    std::atomic<uint64_t> bakedTextureBytes{0};

    void reset() {
      totalDrawCallsProcessed = 0;
      cacheHits = 0;
      cacheMisses = 0;
      stableIdComputations = 0;
      stabilityChecks = 0;
      boundsComputations = 0;
      uvCaptures = 0;
      textureBakes = 0;
      geometryRecordBytes = 0;
      capturedUVBytes = 0;
      bakedTextureBytes = 0;
    }
  };

  /**
   * Result of processing a draw call through the geometry identity system
   */
  struct GeometryIdentityResult {
    XXH64_hash_t stableId;
    GeometryRecord* record;           // Pointer to cached record (valid until cache modified)
    bool isNewGeometry;               // First time seeing this geometry
    bool needsProcessing;             // Requires additional processing (UV capture, bake, etc.)
    bool skipDraw;                    // Should skip this draw call entirely
  };

  /**
   * GeometryIdentityManager
   *
   * Central coordinator for stable geometry identification in shader-based D3D9 games.
   * Handles:
   * - Stable geometry identification (independent of vertex data changes)
   * - Proper culling (frustum-based, not Remix's fixed-function assumptions)
   * - UV capture and correction
   * - VRAM-efficient texture handling
   * - Geometry and texture replacement submission
   *
   * Thread-safety: All public methods are thread-safe via internal locking.
   * Performance: Optimized for minimal per-draw-call overhead (<1μs target).
   */
  class GeometryIdentityManager {
  public:
    GeometryIdentityManager(DxvkDevice* device);
    ~GeometryIdentityManager();

    // Non-copyable
    GeometryIdentityManager(const GeometryIdentityManager&) = delete;
    GeometryIdentityManager& operator=(const GeometryIdentityManager&) = delete;

    /**
     * Initialize the system. Call once at startup.
     */
    void initialize(const GeometryIdentityConfig& config = GeometryIdentityConfig());

    /**
     * Shutdown and release all resources.
     */
    void shutdown();

    /**
     * Called at the start of each frame.
     * Resets per-frame state like draw order counter.
     */
    void onFrameBegin(uint32_t frameIndex);

    /**
     * Called at the end of each frame.
     * Performs deferred processing and cleanup.
     */
    void onFrameEnd();

    /**
     * Process a draw call and return the stable ID and cached record.
     * This is the main entry point called for every draw call.
     *
     * @param draw The draw call state
     * @param vsHash Vertex shader hash
     * @param psHash Pixel shader hash
     * @param textureHashes Array of bound texture hashes
     * @param textureCount Number of bound textures
     * @return Result containing stable ID and processing status
     */
    GeometryIdentityResult processDrawCall(
      const DrawCallState& draw,
      XXH64_hash_t vsHash,
      XXH64_hash_t psHash,
      const XXH64_hash_t* textureHashes,
      uint32_t textureCount
    );

    /**
     * Get or create a geometry record by stable ID.
     * Thread-safe.
     */
    GeometryRecord& getOrCreateRecord(XXH64_hash_t stableId);

    /**
     * Get an existing record or nullptr if not found.
     * Thread-safe.
     */
    GeometryRecord* getRecord(XXH64_hash_t stableId);

    /**
     * Compute world-space bounds for a draw call.
     * Uses vertex sampling for performance.
     */
    AABB computeWorldSpaceBounds(
      const DrawCallState& draw,
      const Matrix4& objectToWorld
    );

    /**
     * Capture UVs from vertex buffer for a draw call.
     */
    CapturedUVData captureUVs(const DrawCallState& draw);

    /**
     * Classify UV behavior based on shader analysis and multi-frame observation.
     */
    UVBehavior classifyUVBehavior(
      const DrawCallState& draw,
      const GeometryRecord& record
    );

    /**
     * Register a replacement geometry for a stable ID.
     */
    void registerReplacementGeometry(
      XXH64_hash_t stableId,
      XXH64_hash_t replacementHash
    );

    /**
     * Register a replacement texture for an original texture.
     */
    void registerReplacementTexture(
      XXH64_hash_t originalHash,
      XXH64_hash_t replacementHash
    );

    /**
     * Get current configuration.
     */
    const GeometryIdentityConfig& getConfig() const { return m_config; }

    /**
     * Get statistics for monitoring.
     */
    const GeometryIdentityStats& getStats() const { return m_stats; }

    /**
     * Reset statistics.
     */
    void resetStats() { m_stats.reset(); }

    /**
     * Get cache size (number of geometry records).
     */
    size_t getCacheSize() const;

    /**
     * Clear all cached records. Use with caution.
     */
    void clearCache();

    /**
     * Check if system is initialized.
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * Prepare geometry submission data for Remix.
     * Combines all processed data into a submission structure.
     */
    GeometrySubmission prepareSubmission(
      const DrawCallState& draw,
      const GeometryRecord& record,
      const Matrix4& objectToWorld
    );

    /**
     * Build culling information for a draw call.
     */
    CullingInfo buildCullingInfo(
      const DrawCallState& draw,
      const AABB& bounds,
      const Matrix4& viewProjection
    );

    /**
     * Check if geometry should be culled based on our frustum check.
     * Returns false if geometry should be visible, true if it can be culled.
     */
    bool shouldCullGeometry(
      const AABB& worldBounds,
      const Frustum& frustum
    );

    /**
     * Load a replacement asset for a stable ID.
     * Returns true if replacement was found and loaded.
     */
    bool loadReplacementAsset(XXH64_hash_t stableId);

    /**
     * Check if a replacement asset exists for a stable ID.
     */
    bool hasReplacementAsset(XXH64_hash_t stableId) const;

  private:
    // Internal methods

    /**
     * Build identity key from draw call state.
     * Must be fast - no vertex data access.
     */
    GeometryIdentityKey buildIdentityKey(
      const DrawCallState& draw,
      XXH64_hash_t vsHash,
      XXH64_hash_t psHash,
      XXH64_hash_t textureBindingsHash
    );

    /**
     * Perform stability check for a geometry (first N frames only).
     */
    void performStabilityCheck(
      GeometryRecord& record,
      const DrawCallState& draw
    );

    /**
     * Sample vertex positions for bounds computation.
     */
    void sampleVertexPositions(
      const DrawCallState& draw,
      std::vector<Vector3>& outPositions,
      uint32_t maxSamples
    );

    /**
     * Read a single vertex position from buffer.
     */
    Vector3 readVertexPosition(
      const DrawCallState& draw,
      uint32_t vertexIndex
    );

    /**
     * Read a single UV coordinate from buffer.
     */
    Vector2 readVertexUV(
      const DrawCallState& draw,
      uint32_t vertexIndex
    );

    /**
     * Hash vertex data for stability detection.
     * Expensive - only called during stability check phase.
     */
    XXH64_hash_t hashVertexData(const DrawCallState& draw);

    // Member variables
    DxvkDevice* m_device;
    GeometryIdentityConfig m_config;
    GeometryIdentityStats m_stats;
    bool m_initialized;

    // Geometry cache - maps stable ID to record
    std::unordered_map<XXH64_hash_t, GeometryRecord> m_geometryCache;
    mutable std::mutex m_cacheMutex;

    // Texture replacements - maps original hash to replacement hash
    std::unordered_map<XXH64_hash_t, XXH64_hash_t> m_textureReplacements;
    mutable std::mutex m_textureReplacementMutex;

    // Per-frame state
    std::atomic<uint32_t> m_currentFrame{0};
    std::atomic<uint32_t> m_drawOrderCounter{0};

    // Deferred processing queue
    std::vector<XXH64_hash_t> m_deferredProcessingQueue;
    std::mutex m_deferredQueueMutex;
  };

} // namespace dxvk
