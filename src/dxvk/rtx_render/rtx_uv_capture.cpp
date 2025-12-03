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

#include "rtx_uv_capture.h"
#include "rtx_types.h"
#include "../util/log/log.h"

#include <algorithm>
#include <unordered_set>

namespace dxvk {

  void UVCaptureManager::initialize(const UVCaptureConfig& config) {
    m_config = config;

    // Pre-allocate caches
    m_uvCache.reserve(m_config.maxCachedCaptures);
    m_behaviorTrackers.reserve(1024);
  }

  void UVCaptureManager::shutdown() {
    clearCache();
  }

  UVCaptureData UVCaptureManager::captureUVs(
    const DrawCallState& draw,
    XXH64_hash_t stableId,
    uint32_t currentFrame
  ) {
    // Check cache first
    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);
      auto it = m_uvCache.find(stableId);
      if (it != m_uvCache.end() && it->second.isComplete) {
        return it->second;
      }
    }

    // Capture new UV data
    UVCaptureData result;
    result.geometryStableId = stableId;
    result.captureFrame = currentFrame;

    // Read UVs from vertex buffer
    result.uvs = readUVsFromBuffer(draw, 0);
    result.isComplete = !result.uvs.empty();

    // Optionally capture second UV set
    if (m_config.captureSecondUVSet) {
      result.uvs2 = readUVsFromBuffer(draw, 1);
      result.hasSecondUVSet = !result.uvs2.empty();
    }

    // Determine behavior
    if (result.isComplete) {
      // Check if we have behavior tracking data
      UVBehavior behavior = getDetectedBehavior(stableId);
      if (behavior != UVBehavior::Unknown) {
        result.behavior = behavior;
      } else {
        // Default to static for now, tracking will refine this
        result.behavior = UVBehavior::Static;
      }
    }

    // Cache the result
    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);

      // Evict oldest if at capacity
      if (m_uvCache.size() >= m_config.maxCachedCaptures) {
        // Simple eviction: remove first entry (could be improved with LRU)
        m_uvCache.erase(m_uvCache.begin());
      }

      m_uvCache[stableId] = result;
    }

    return result;
  }

  const UVCaptureData* UVCaptureManager::getCachedUVs(XXH64_hash_t stableId) const {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_uvCache.find(stableId);
    if (it != m_uvCache.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void UVCaptureManager::trackUVBehavior(
    XXH64_hash_t stableId,
    const std::vector<Vector2>& uvs,
    uint32_t currentFrame
  ) {
    if (!m_config.detectAnimatedUVs || uvs.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(m_trackerMutex);

    auto& tracker = m_behaviorTrackers[stableId];
    if (tracker.detectionComplete) {
      return;  // Already determined behavior
    }

    tracker.geometryStableId = stableId;

    // Compute UV hash and bounds
    XXH64_hash_t uvHash = computeUVHash(uvs);
    Vector2 minUV, maxUV;
    computeUVBounds(uvs, minUV, maxUV);

    tracker.updateWithUVHash(uvHash, minUV, maxUV);

    // Check if we have enough data to finalize
    if (tracker.framesObserved >= m_config.animationDetectionFrames) {
      UVBehavior behavior = tracker.analyzeAndFinalize(m_config.animationDetectionFrames);

      // Update cached UV data with detected behavior
      std::lock_guard<std::mutex> cacheLock(m_cacheMutex);
      auto it = m_uvCache.find(stableId);
      if (it != m_uvCache.end()) {
        it->second.behavior = behavior;
      }
    }
  }

  UVBehavior UVCaptureManager::getDetectedBehavior(XXH64_hash_t stableId) const {
    std::lock_guard<std::mutex> lock(m_trackerMutex);

    auto it = m_behaviorTrackers.find(stableId);
    if (it != m_behaviorTrackers.end() && it->second.detectionComplete) {
      return it->second.detectedBehavior;
    }
    return UVBehavior::Unknown;
  }

  AnimatedUVParams UVCaptureManager::extractAnimatedParams(
    const DrawCallState& draw,
    XXH64_hash_t stableId
  ) {
    AnimatedUVParams params;

    // TODO: Analyze shader constants to detect animation parameters
    // This would involve:
    // 1. Looking at VS constants that affect texture coordinates
    // 2. Detecting time-based constants
    // 3. Detecting scroll rate constants
    // 4. Detecting rotation constants

    // For now, return empty params - animation detection is done via
    // multi-frame UV hash tracking in trackUVBehavior()

    return params;
  }

  void UVCaptureManager::clearCacheEntry(XXH64_hash_t stableId) {
    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);
      m_uvCache.erase(stableId);
    }
    {
      std::lock_guard<std::mutex> lock(m_trackerMutex);
      m_behaviorTrackers.erase(stableId);
    }
  }

  void UVCaptureManager::clearCache() {
    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);
      m_uvCache.clear();
    }
    {
      std::lock_guard<std::mutex> lock(m_trackerMutex);
      m_behaviorTrackers.clear();
    }
  }

  size_t UVCaptureManager::getTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    size_t total = 0;
    for (const auto& [id, data] : m_uvCache) {
      total += data.getMemoryUsage();
    }
    return total;
  }

  std::vector<Vector2> UVCaptureManager::readUVsFromBuffer(
    const DrawCallState& draw,
    uint32_t uvSetIndex
  ) {
    std::vector<Vector2> uvs;

    const auto& geometryData = draw.getGeometryData();

    // Select appropriate texcoord buffer based on UV set index
    const auto& texcoordBuffer = (uvSetIndex == 0) ?
      geometryData.texcoordBuffer :
      geometryData.texcoordBuffer;  // TODO: Support multiple UV sets

    if (!texcoordBuffer.defined()) {
      return uvs;
    }

    const auto bufferSlice = texcoordBuffer.getSliceHandle();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(bufferSlice.mapPtr);

    if (data == nullptr) {
      return uvs;
    }

    uint32_t stride = texcoordBuffer.stride();
    uint32_t vertexCount = geometryData.vertexCount;

    uvs.reserve(vertexCount);

    for (uint32_t i = 0; i < vertexCount; i++) {
      uint32_t offset = i * stride;
      const float* uv = reinterpret_cast<const float*>(data + offset);
      uvs.push_back(Vector2(uv[0], uv[1]));
    }

    return uvs;
  }

  XXH64_hash_t UVCaptureManager::computeUVHash(const std::vector<Vector2>& uvs) {
    if (uvs.empty()) {
      return 0;
    }
    return XXH3_64bits(uvs.data(), uvs.size() * sizeof(Vector2));
  }

  void UVCaptureManager::computeUVBounds(
    const std::vector<Vector2>& uvs,
    Vector2& outMin,
    Vector2& outMax
  ) {
    outMin = Vector2(FLT_MAX, FLT_MAX);
    outMax = Vector2(-FLT_MAX, -FLT_MAX);

    for (const auto& uv : uvs) {
      outMin.x = std::min(outMin.x, uv.x);
      outMin.y = std::min(outMin.y, uv.y);
      outMax.x = std::max(outMax.x, uv.x);
      outMax.y = std::max(outMax.y, uv.y);
    }
  }

} // namespace dxvk
