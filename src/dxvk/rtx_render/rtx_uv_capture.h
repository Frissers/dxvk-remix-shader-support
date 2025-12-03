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

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "../../util/xxHash/xxhash.h"
#include "../util/util_vector.h"
#include "rtx_geometry_identity.h"

namespace dxvk {

  struct DrawCallState;

  /**
   * Animated UV parameters extracted from shader analysis
   */
  struct AnimatedUVParams {
    Vector2 scrollRate;      // UV scroll rate (units per second)
    Vector2 scrollOffset;    // Current scroll offset
    float rotationRate;      // Rotation rate (radians per second)
    float rotationAngle;     // Current rotation angle
    float scaleOscillation;  // Scale oscillation amplitude
    float scaleFrequency;    // Scale oscillation frequency

    bool hasScroll;
    bool hasRotation;
    bool hasScaleAnimation;

    AnimatedUVParams()
      : scrollRate(0.0f, 0.0f)
      , scrollOffset(0.0f, 0.0f)
      , rotationRate(0.0f)
      , rotationAngle(0.0f)
      , scaleOscillation(0.0f)
      , scaleFrequency(0.0f)
      , hasScroll(false)
      , hasRotation(false)
      , hasScaleAnimation(false) {}

    bool isAnimated() const {
      return hasScroll || hasRotation || hasScaleAnimation;
    }
  };

  /**
   * UV capture result with full metadata
   */
  struct UVCaptureData {
    XXH64_hash_t geometryStableId;
    std::vector<Vector2> uvs;         // Per-vertex UVs (UV set 0)
    std::vector<Vector2> uvs2;        // Per-vertex UVs (UV set 1, if available)
    UVBehavior behavior;
    AnimatedUVParams animatedParams;
    uint32_t captureFrame;            // Frame when captured
    bool isComplete;
    bool hasSecondUVSet;

    UVCaptureData()
      : geometryStableId(0)
      , behavior(UVBehavior::Unknown)
      , captureFrame(0)
      , isComplete(false)
      , hasSecondUVSet(false) {}

    size_t getMemoryUsage() const {
      return uvs.size() * sizeof(Vector2) +
             uvs2.size() * sizeof(Vector2) +
             sizeof(UVCaptureData);
    }
  };

  /**
   * UV capture system configuration
   */
  struct UVCaptureConfig {
    bool enableCapture = true;
    bool detectAnimatedUVs = true;
    bool captureSecondUVSet = false;
    uint32_t maxCachedCaptures = 4096;
    uint32_t animationDetectionFrames = 16;  // Frames to observe for animation detection
  };

  /**
   * UV behavior tracking for animation detection
   */
  struct UVBehaviorTracker {
    XXH64_hash_t geometryStableId;
    uint32_t framesObserved;
    std::vector<XXH64_hash_t> uvHashes;  // Hash of UV data per frame
    Vector2 minUVSeen;
    Vector2 maxUVSeen;
    bool detectionComplete;
    UVBehavior detectedBehavior;

    UVBehaviorTracker()
      : geometryStableId(0)
      , framesObserved(0)
      , minUVSeen(FLT_MAX, FLT_MAX)
      , maxUVSeen(-FLT_MAX, -FLT_MAX)
      , detectionComplete(false)
      , detectedBehavior(UVBehavior::Unknown) {}

    void updateWithUVHash(XXH64_hash_t hash, const Vector2& minUV, const Vector2& maxUV) {
      uvHashes.push_back(hash);
      minUVSeen.x = std::min(minUVSeen.x, minUV.x);
      minUVSeen.y = std::min(minUVSeen.y, minUV.y);
      maxUVSeen.x = std::max(maxUVSeen.x, maxUV.x);
      maxUVSeen.y = std::max(maxUVSeen.y, maxUV.y);
      framesObserved++;
    }

    UVBehavior analyzeAndFinalize(uint32_t requiredFrames) {
      if (framesObserved < requiredFrames) {
        return UVBehavior::Unknown;
      }

      detectionComplete = true;

      // Check how many unique hashes we've seen
      std::unordered_set<XXH64_hash_t> uniqueHashes(uvHashes.begin(), uvHashes.end());

      if (uniqueHashes.size() == 1) {
        // UVs never changed
        detectedBehavior = UVBehavior::Static;
      } else if (uniqueHashes.size() <= 2) {
        // Minor variations, likely numerical precision
        detectedBehavior = UVBehavior::Transformed;
      } else {
        // UVs are changing - could be animated or view-dependent
        // Check UV range for scrolling pattern
        float uRange = maxUVSeen.x - minUVSeen.x;
        float vRange = maxUVSeen.y - minUVSeen.y;

        if (uRange > 2.0f || vRange > 2.0f) {
          // Large UV range suggests scrolling/animated
          detectedBehavior = UVBehavior::Animated;
        } else {
          // Could be procedural or view-dependent
          detectedBehavior = UVBehavior::Procedural;
        }
      }

      return detectedBehavior;
    }
  };

  /**
   * UVCaptureManager
   *
   * Handles UV capture, caching, and animation detection.
   */
  class UVCaptureManager {
  public:
    UVCaptureManager() = default;
    ~UVCaptureManager() = default;

    void initialize(const UVCaptureConfig& config = UVCaptureConfig());
    void shutdown();

    /**
     * Capture UVs for a draw call
     */
    UVCaptureData captureUVs(
      const DrawCallState& draw,
      XXH64_hash_t stableId,
      uint32_t currentFrame
    );

    /**
     * Get cached UV data for a geometry
     */
    const UVCaptureData* getCachedUVs(XXH64_hash_t stableId) const;

    /**
     * Track UV behavior over multiple frames for animation detection
     */
    void trackUVBehavior(
      XXH64_hash_t stableId,
      const std::vector<Vector2>& uvs,
      uint32_t currentFrame
    );

    /**
     * Get detected UV behavior for a geometry
     */
    UVBehavior getDetectedBehavior(XXH64_hash_t stableId) const;

    /**
     * Extract animated UV parameters from shader constants
     */
    AnimatedUVParams extractAnimatedParams(
      const DrawCallState& draw,
      XXH64_hash_t stableId
    );

    /**
     * Clear UV cache for a specific geometry
     */
    void clearCacheEntry(XXH64_hash_t stableId);

    /**
     * Clear all cached data
     */
    void clearCache();

    /**
     * Get memory usage statistics
     */
    size_t getTotalMemoryUsage() const;

  private:
    UVCaptureConfig m_config;

    // Cached UV captures
    std::unordered_map<XXH64_hash_t, UVCaptureData> m_uvCache;
    mutable std::mutex m_cacheMutex;

    // Behavior tracking for animation detection
    std::unordered_map<XXH64_hash_t, UVBehaviorTracker> m_behaviorTrackers;
    mutable std::mutex m_trackerMutex;

    // Helper to read UVs from draw call
    std::vector<Vector2> readUVsFromBuffer(
      const DrawCallState& draw,
      uint32_t uvSetIndex
    );

    // Compute hash of UV data for tracking
    XXH64_hash_t computeUVHash(const std::vector<Vector2>& uvs);

    // Get UV bounds
    void computeUVBounds(
      const std::vector<Vector2>& uvs,
      Vector2& outMin,
      Vector2& outMax
    );
  };

} // namespace dxvk
