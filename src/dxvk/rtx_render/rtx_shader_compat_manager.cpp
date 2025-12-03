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

#include "rtx_shader_compat_manager.h"
#include "rtx_types.h"
#include "../util/log/log.h"

#include <algorithm>

namespace dxvk {

  GeometryIdentityManager::GeometryIdentityManager(DxvkDevice* device)
    : m_device(device)
    , m_initialized(false) {
  }

  GeometryIdentityManager::~GeometryIdentityManager() {
    shutdown();
  }

  void GeometryIdentityManager::initialize(const GeometryIdentityConfig& config) {
    if (m_initialized) return;

    m_config = config;

    // Pre-allocate cache with reasonable initial capacity
    m_geometryCache.reserve(4096);
    m_textureReplacements.reserve(1024);
    m_deferredProcessingQueue.reserve(256);

    m_initialized = true;

    if (m_config.enableDebugLogging) {
      Logger::info("[ShaderCompat] Initialized with config:");
      Logger::info(str::format("  stabilityCheckFrames: ", config.stabilityCheckFrames));
      Logger::info(str::format("  boundsSampleCount: ", config.boundsSampleCount));
      Logger::info(str::format("  enableUVCapture: ", config.enableUVCapture));
      Logger::info(str::format("  enableTextureBaking: ", config.enableTextureBaking));
      Logger::info(str::format("  useMemoryAliasing: ", config.useMemoryAliasing));
    }
  }

  void GeometryIdentityManager::shutdown() {
    if (!m_initialized) return;

    clearCache();
    m_initialized = false;

    if (m_config.enableDebugLogging) {
      Logger::info("[ShaderCompat] Shutdown complete");
    }
  }

  void GeometryIdentityManager::onFrameBegin(uint32_t frameIndex) {
    m_currentFrame.store(frameIndex, std::memory_order_relaxed);
    m_drawOrderCounter.store(0, std::memory_order_relaxed);
  }

  void GeometryIdentityManager::onFrameEnd() {
    // Process deferred queue
    std::vector<XXH64_hash_t> toProcess;
    {
      std::lock_guard<std::mutex> lock(m_deferredQueueMutex);
      toProcess.swap(m_deferredProcessingQueue);
    }

    // TODO: Process deferred items (UV capture, texture baking, etc.)
    // This is done outside the main draw loop for better batching

    if (m_config.enableDebugLogging && !toProcess.empty()) {
      Logger::info(str::format("[ShaderCompat] Frame end: processed ",
                              toProcess.size(), " deferred items"));
    }
  }

  GeometryIdentityResult GeometryIdentityManager::processDrawCall(
    const DrawCallState& draw,
    XXH64_hash_t vsHash,
    XXH64_hash_t psHash,
    const XXH64_hash_t* textureHashes,
    uint32_t textureCount
  ) {
    m_stats.totalDrawCallsProcessed++;

    // Step 1: Compute texture bindings hash
    XXH64_hash_t textureBindingsHash = GeometryIdentityKey::computeTextureBindingsHash(
      textureHashes, textureCount
    );

    // Step 2: Build identity key (fast - no vertex data access)
    GeometryIdentityKey key = buildIdentityKey(draw, vsHash, psHash, textureBindingsHash);

    // Step 3: Compute stable ID
    XXH64_hash_t stableId = key.computeStableId();
    m_stats.stableIdComputations++;

    // Step 4: Cache lookup
    GeometryIdentityResult result;
    result.stableId = stableId;
    result.skipDraw = false;

    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);

      auto it = m_geometryCache.find(stableId);
      if (it != m_geometryCache.end()) {
        // Cache hit
        m_stats.cacheHits++;
        result.record = &it->second;
        result.isNewGeometry = false;
      } else {
        // Cache miss - create new record
        m_stats.cacheMisses++;

        auto [newIt, inserted] = m_geometryCache.emplace(stableId, GeometryRecord());
        result.record = &newIt->second;
        result.record->stableId = stableId;
        result.isNewGeometry = true;

        m_stats.geometryRecordBytes += sizeof(GeometryRecord);
      }

      // Update last seen frame
      result.record->lastSeenFrame = m_currentFrame.load(std::memory_order_relaxed);
    }

    // Step 5: Stability check (first N frames only)
    if (!result.record->stabilityCheckComplete) {
      performStabilityCheck(*result.record, draw);
    }

    // Step 6: Determine if processing is needed
    result.needsProcessing = result.record->needsProcessing();

    // Step 7: Queue deferred processing if needed
    if (result.needsProcessing && result.isNewGeometry) {
      std::lock_guard<std::mutex> lock(m_deferredQueueMutex);
      m_deferredProcessingQueue.push_back(stableId);
    }

    return result;
  }

  GeometryIdentityKey GeometryIdentityManager::buildIdentityKey(
    const DrawCallState& draw,
    XXH64_hash_t vsHash,
    XXH64_hash_t psHash,
    XXH64_hash_t textureBindingsHash
  ) {
    GeometryIdentityKey key;
    key.vsHash = vsHash;
    key.psHash = psHash;
    key.textureBindingsHash = textureBindingsHash;

    // Get geometry info from draw call
    const auto& geometryData = draw.getGeometryData();
    key.vertexCount = geometryData.vertexCount;
    key.indexCount = geometryData.indexCount;

    // Get vertex stride from first position buffer
    const auto& positionBuffer = geometryData.positionBuffer;
    key.vertexStride = positionBuffer.stride();

    // Primitive topology
    key.primitiveTopology = static_cast<uint32_t>(geometryData.topology);

    // Draw order (atomic increment)
    key.drawOrderIndex = m_drawOrderCounter.fetch_add(1, std::memory_order_relaxed);

    key.padding = 0;

    return key;
  }

  GeometryRecord& GeometryIdentityManager::getOrCreateRecord(XXH64_hash_t stableId) {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_geometryCache.find(stableId);
    if (it != m_geometryCache.end()) {
      return it->second;
    }

    auto [newIt, inserted] = m_geometryCache.emplace(stableId, GeometryRecord());
    newIt->second.stableId = stableId;
    m_stats.geometryRecordBytes += sizeof(GeometryRecord);
    return newIt->second;
  }

  GeometryRecord* GeometryIdentityManager::getRecord(XXH64_hash_t stableId) {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_geometryCache.find(stableId);
    if (it != m_geometryCache.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void GeometryIdentityManager::performStabilityCheck(
    GeometryRecord& record,
    const DrawCallState& draw
  ) {
    if (record.stabilityCheckComplete) return;

    m_stats.stabilityChecks++;

    // Hash vertex data (expensive - only during detection)
    XXH64_hash_t vertexHash = hashVertexData(draw);

    // Update tracking
    record.updateStabilityTracking(vertexHash, m_currentFrame.load(std::memory_order_relaxed));

    if (record.stabilityCheckComplete && m_config.enableDebugLogging) {
      Logger::info(str::format("[ShaderCompat] StableID 0x", std::hex, record.stableId, std::dec,
                              " stability check complete: ",
                              record.isStable ? "STABLE" : "DYNAMIC",
                              " (variations: ", record.hashVariations, ")"));
    }
  }

  XXH64_hash_t GeometryIdentityManager::hashVertexData(const DrawCallState& draw) {
    const auto& geometryData = draw.getGeometryData();
    const auto& positionBuffer = geometryData.positionBuffer;

    if (!positionBuffer.defined()) {
      return 0;
    }

    // Get buffer slice info
    const auto bufferSlice = positionBuffer.getSliceHandle();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(bufferSlice.mapPtr);

    if (data == nullptr) {
      return 0;
    }

    // Hash position data
    uint32_t stride = positionBuffer.stride();
    uint32_t vertexCount = geometryData.vertexCount;
    size_t dataSize = static_cast<size_t>(stride) * vertexCount;

    return XXH3_64bits(data, dataSize);
  }

  AABB GeometryIdentityManager::computeWorldSpaceBounds(
    const DrawCallState& draw,
    const Matrix4& objectToWorld
  ) {
    m_stats.boundsComputations++;

    AABB bounds = AABB::empty();

    const auto& geometryData = draw.getGeometryData();
    uint32_t vertexCount = geometryData.vertexCount;

    if (vertexCount == 0) return bounds;

    // Sample vertices for bounds (don't need all of them)
    uint32_t sampleCount = std::min(vertexCount, m_config.boundsSampleCount);
    uint32_t stride = std::max(1u, vertexCount / sampleCount);

    for (uint32_t i = 0; i < vertexCount; i += stride) {
      Vector3 localPos = readVertexPosition(draw, i);
      Vector4 worldPos4 = objectToWorld * Vector4(localPos.x, localPos.y, localPos.z, 1.0f);
      Vector3 worldPos(worldPos4.x, worldPos4.y, worldPos4.z);
      bounds.expand(worldPos);
    }

    return bounds;
  }

  Vector3 GeometryIdentityManager::readVertexPosition(
    const DrawCallState& draw,
    uint32_t vertexIndex
  ) {
    const auto& geometryData = draw.getGeometryData();
    const auto& positionBuffer = geometryData.positionBuffer;

    if (!positionBuffer.defined()) {
      return Vector3(0.0f, 0.0f, 0.0f);
    }

    const auto bufferSlice = positionBuffer.getSliceHandle();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(bufferSlice.mapPtr);

    if (data == nullptr) {
      return Vector3(0.0f, 0.0f, 0.0f);
    }

    uint32_t stride = positionBuffer.stride();
    uint32_t offset = vertexIndex * stride;

    // Assume position is at offset 0 and is 3 floats (common D3D9 format)
    const float* pos = reinterpret_cast<const float*>(data + offset);

    return Vector3(pos[0], pos[1], pos[2]);
  }

  Vector2 GeometryIdentityManager::readVertexUV(
    const DrawCallState& draw,
    uint32_t vertexIndex
  ) {
    const auto& geometryData = draw.getGeometryData();
    const auto& texcoordBuffer = geometryData.texcoordBuffer;

    if (!texcoordBuffer.defined()) {
      return Vector2(0.0f, 0.0f);
    }

    const auto bufferSlice = texcoordBuffer.getSliceHandle();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(bufferSlice.mapPtr);

    if (data == nullptr) {
      return Vector2(0.0f, 0.0f);
    }

    uint32_t stride = texcoordBuffer.stride();
    uint32_t offset = vertexIndex * stride;

    // Assume UV is at offset 0 and is 2 floats
    const float* uv = reinterpret_cast<const float*>(data + offset);

    return Vector2(uv[0], uv[1]);
  }

  CapturedUVData GeometryIdentityManager::captureUVs(const DrawCallState& draw) {
    m_stats.uvCaptures++;

    CapturedUVData result;
    result.behavior = UVBehavior::Unknown;
    result.isComplete = false;

    const auto& geometryData = draw.getGeometryData();
    uint32_t vertexCount = geometryData.vertexCount;

    if (vertexCount == 0) return result;

    result.uvs.reserve(vertexCount);

    for (uint32_t i = 0; i < vertexCount; i++) {
      Vector2 uv = readVertexUV(draw, i);
      result.uvs.push_back(uv);
    }

    result.isComplete = (result.uvs.size() == vertexCount);

    // Track memory
    m_stats.capturedUVBytes += result.uvs.size() * sizeof(Vector2);

    return result;
  }

  UVBehavior GeometryIdentityManager::classifyUVBehavior(
    const DrawCallState& draw,
    const GeometryRecord& record
  ) {
    // Simple classification for now
    // TODO: Analyze shader to detect animated/procedural UVs

    if (!record.isStable) {
      // Geometry changes, UVs might too
      return UVBehavior::Procedural;
    }

    // Default to static for stable geometry
    return UVBehavior::Static;
  }

  void GeometryIdentityManager::registerReplacementGeometry(
    XXH64_hash_t stableId,
    XXH64_hash_t replacementHash
  ) {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_geometryCache.find(stableId);
    if (it != m_geometryCache.end()) {
      it->second.hasReplacementGeometry = true;
      it->second.replacementGeometryHash = replacementHash;
    }
  }

  void GeometryIdentityManager::registerReplacementTexture(
    XXH64_hash_t originalHash,
    XXH64_hash_t replacementHash
  ) {
    std::lock_guard<std::mutex> lock(m_textureReplacementMutex);
    m_textureReplacements[originalHash] = replacementHash;
  }

  size_t GeometryIdentityManager::getCacheSize() const {
    std::lock_guard<std::mutex> lock(m_cacheMutex);
    return m_geometryCache.size();
  }

  void GeometryIdentityManager::clearCache() {
    {
      std::lock_guard<std::mutex> lock(m_cacheMutex);
      m_geometryCache.clear();
    }

    {
      std::lock_guard<std::mutex> lock(m_textureReplacementMutex);
      m_textureReplacements.clear();
    }

    {
      std::lock_guard<std::mutex> lock(m_deferredQueueMutex);
      m_deferredProcessingQueue.clear();
    }

    // Reset memory stats
    m_stats.geometryRecordBytes = 0;
    m_stats.capturedUVBytes = 0;
    m_stats.bakedTextureBytes = 0;
  }

  void GeometryIdentityManager::sampleVertexPositions(
    const DrawCallState& draw,
    std::vector<Vector3>& outPositions,
    uint32_t maxSamples
  ) {
    const auto& geometryData = draw.getGeometryData();
    uint32_t vertexCount = geometryData.vertexCount;

    if (vertexCount == 0) return;

    // Calculate sampling stride
    uint32_t sampleCount = std::min(vertexCount, maxSamples);
    uint32_t stride = std::max(1u, vertexCount / sampleCount);

    outPositions.clear();
    outPositions.reserve(sampleCount);

    for (uint32_t i = 0; i < vertexCount; i += stride) {
      Vector3 pos = readVertexPosition(draw, i);
      outPositions.push_back(pos);
    }
  }

  GeometrySubmission GeometryIdentityManager::prepareSubmission(
    const DrawCallState& draw,
    const GeometryRecord& record,
    const Matrix4& objectToWorld
  ) {
    GeometrySubmission submission;
    submission.stableId = record.stableId;

    const auto& geometryData = draw.getGeometryData();

    // Vertex data
    const auto& positionBuffer = geometryData.positionBuffer;
    if (positionBuffer.defined()) {
      const auto bufferSlice = positionBuffer.getSliceHandle();
      submission.vertexData = bufferSlice.mapPtr;
      submission.vertexCount = geometryData.vertexCount;
      submission.vertexStride = positionBuffer.stride();
    }

    // Index data
    const auto& indexBuffer = geometryData.indexBuffer;
    if (indexBuffer.defined()) {
      const auto bufferSlice = indexBuffer.getSliceHandle();
      submission.indexData = bufferSlice.mapPtr;
      submission.indexCount = geometryData.indexCount;
      submission.indexStride = sizeof(uint16_t);  // Common for D3D9
    }

    // UV data - use captured UVs if available
    if (!record.capturedUVs.uvs.empty() && record.capturedUVs.isComplete) {
      submission.uvData = record.capturedUVs.uvs.data();
      submission.uvCount = static_cast<uint32_t>(record.capturedUVs.uvs.size());
      submission.uvsAreOverridden = true;
    } else {
      submission.uvData = nullptr;
      submission.uvCount = 0;
      submission.uvsAreOverridden = false;
    }

    // Texture - check for replacement
    if (record.hasReplacementTexture) {
      submission.textureHash = record.replacementTextureHash;
      submission.textureIsReplacement = true;
    } else {
      submission.textureHash = record.originalTextureHash;
      submission.textureIsReplacement = false;
    }

    // Compute world-space bounds
    if (record.boundsValid) {
      submission.worldSpaceBounds = record.worldSpaceBounds;
    } else {
      submission.worldSpaceBounds = computeWorldSpaceBounds(draw, objectToWorld);
    }

    // Transforms
    submission.objectToWorld = objectToWorld;

    // Flags
    submission.hasReplacementAsset = record.hasReplacementGeometry || record.hasReplacementTexture;
    submission.skipRemixProcessing = false;
    submission.forceVisible = record.hasCullingIssues;

    return submission;
  }

  CullingInfo GeometryIdentityManager::buildCullingInfo(
    const DrawCallState& draw,
    const AABB& bounds,
    const Matrix4& viewProjection
  ) {
    CullingInfo info;
    info.worldSpaceBounds = bounds;
    info.viewProjection = viewProjection;
    info.frustum = Frustum::fromViewProjection(viewProjection);
    info.boundsAreAccurate = bounds.isValid();
    info.disableRemixCulling = m_config.overrideRemixCulling;
    info.useGameCullingOnly = m_config.useFrustumCullingOnly;

    return info;
  }

  bool GeometryIdentityManager::shouldCullGeometry(
    const AABB& worldBounds,
    const Frustum& frustum
  ) {
    // If bounds are invalid, don't cull (be conservative)
    if (!worldBounds.isValid()) {
      return false;
    }

    // Use frustum intersection test
    // Returns true if should be culled (i.e., NOT visible)
    return !frustum.intersectsAABB(worldBounds);
  }

  bool GeometryIdentityManager::loadReplacementAsset(XXH64_hash_t stableId) {
    // TODO: Implement replacement asset loading from disk/USD
    // This would load .usd files with matching stable ID hashes
    //
    // Path convention could be:
    //   rtx-remix/mods/replacements/geometry_0xHASH.usd
    //   rtx-remix/mods/replacements/texture_0xHASH.dds
    //
    // For now, just check if a replacement has been registered
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_geometryCache.find(stableId);
    if (it != m_geometryCache.end()) {
      // Check if already has replacement
      if (it->second.hasReplacementGeometry || it->second.hasReplacementTexture) {
        it->second.isFullyProcessed = true;
        return true;
      }
    }

    // No replacement found
    return false;
  }

  bool GeometryIdentityManager::hasReplacementAsset(XXH64_hash_t stableId) const {
    std::lock_guard<std::mutex> lock(m_cacheMutex);

    auto it = m_geometryCache.find(stableId);
    if (it != m_geometryCache.end()) {
      return it->second.hasReplacementGeometry || it->second.hasReplacementTexture;
    }

    return false;
  }

} // namespace dxvk
