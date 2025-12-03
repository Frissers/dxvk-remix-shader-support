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

#include <cstdint>
#include <vector>
#include <cfloat>
#include "../../util/xxHash/xxhash.h"
#include "../util/util_vector.h"
#include "../util/util_matrix.h"

namespace dxvk {

  // Number of frames to track for stability detection
  constexpr uint32_t kStabilityCheckFrames = 8;

  // Maximum hash variations allowed to still be considered "stable"
  constexpr uint32_t kMaxStableVariations = 1;

  /**
   * Geometry type classification
   */
  enum class GeometryType : uint8_t {
    Unknown = 0,
    Static,           // Vertex data never changes
    Skinned,          // Animated via bone transforms
    Morphing,         // Blend shapes / vertex animation
    Procedural,       // Vertices computed in shader
    Dynamic           // Changes unpredictably
  };

  /**
   * UV behavior classification
   */
  enum class UVBehavior : uint8_t {
    Unknown = 0,
    Static,           // UV from vertex buffer, no transformation
    Transformed,      // UV modified by VS but result is static
    Animated,         // UV changes over time (scrolling water, etc.)
    ViewDependent,    // UV depends on camera (rare)
    Procedural        // UV computed entirely in shader
  };

  /**
   * Texture type classification for VRAM-efficient handling
   */
  enum class TextureType : uint8_t {
    Unknown = 0,
    Static,           // Normal texture, passthrough
    RenderTarget,     // Game renders to this texture
    Procedural,       // Generated entirely by pixel shader
    Animated          // Texture content changes (video, etc.)
  };

  /**
   * Lightweight identity key - computed every frame per draw call.
   * Must be FAST - no vertex data access, just existing state.
   *
   * Size: 48 bytes (fits in cache line)
   */
  struct GeometryIdentityKey {
    XXH64_hash_t vsHash;              // Vertex shader hash
    XXH64_hash_t psHash;              // Pixel shader hash
    XXH64_hash_t textureBindingsHash; // Combined hash of bound texture hashes
    uint32_t vertexCount;             // Number of vertices
    uint32_t indexCount;              // Number of indices (0 if non-indexed)
    uint32_t vertexStride;            // Bytes per vertex
    uint32_t primitiveTopology;       // D3D primitive type
    uint32_t drawOrderIndex;          // Nth draw call this frame (disambiguates)
    uint32_t padding;                 // Alignment padding

    /**
     * Compute stable ID using XXH3 (faster than XXH64 for small inputs)
     * This is the primary identifier used throughout the system.
     */
    XXH64_hash_t computeStableId() const {
      // Structure is already packed and aligned for efficient hashing
      return XXH3_64bits(this, sizeof(GeometryIdentityKey));
    }

    /**
     * Compute hash of bound textures for the identity key
     */
    static XXH64_hash_t computeTextureBindingsHash(
      const XXH64_hash_t* textureHashes,
      uint32_t count
    ) {
      if (count == 0) return 0;
      return XXH3_64bits(textureHashes, count * sizeof(XXH64_hash_t));
    }

    bool operator==(const GeometryIdentityKey& other) const {
      return vsHash == other.vsHash &&
             psHash == other.psHash &&
             textureBindingsHash == other.textureBindingsHash &&
             vertexCount == other.vertexCount &&
             indexCount == other.indexCount &&
             vertexStride == other.vertexStride &&
             primitiveTopology == other.primitiveTopology &&
             drawOrderIndex == other.drawOrderIndex;
    }
  };

  static_assert(sizeof(GeometryIdentityKey) == 48, "GeometryIdentityKey should be 48 bytes");

  /**
   * Axis-aligned bounding box for culling
   */
  struct AABB {
    Vector3 minBounds;
    Vector3 maxBounds;

    static AABB empty() {
      AABB aabb;
      aabb.minBounds = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);
      aabb.maxBounds = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      return aabb;
    }

    void expand(const Vector3& point) {
      minBounds.x = std::min(minBounds.x, point.x);
      minBounds.y = std::min(minBounds.y, point.y);
      minBounds.z = std::min(minBounds.z, point.z);
      maxBounds.x = std::max(maxBounds.x, point.x);
      maxBounds.y = std::max(maxBounds.y, point.y);
      maxBounds.z = std::max(maxBounds.z, point.z);
    }

    Vector3 center() const {
      return (minBounds + maxBounds) * 0.5f;
    }

    Vector3 extents() const {
      return (maxBounds - minBounds) * 0.5f;
    }

    bool isValid() const {
      return minBounds.x <= maxBounds.x &&
             minBounds.y <= maxBounds.y &&
             minBounds.z <= maxBounds.z;
    }
  };

  /**
   * Frustum planes for culling
   * Plane equation: ax + by + cz + d = 0 where (a,b,c) is normal and d is distance
   * Point is on positive side if ax + by + cz + d > 0
   */
  struct Frustum {
    Vector4 planes[6];  // Left, Right, Bottom, Top, Near, Far

    /**
     * Extract frustum planes from a view-projection matrix.
     * Uses Gribb/Hartmann method for extracting planes from clip matrix.
     */
    static Frustum fromViewProjection(const Matrix4& viewProj) {
      Frustum f;

      // In row-major matrix convention (DirectX style):
      // Row 0 = (m00, m01, m02, m03)
      // Row 1 = (m10, m11, m12, m13)
      // Row 2 = (m20, m21, m22, m23)
      // Row 3 = (m30, m31, m32, m33)

      // Left plane: row 4 + row 1
      f.planes[0] = Vector4(
        viewProj[0][3] + viewProj[0][0],
        viewProj[1][3] + viewProj[1][0],
        viewProj[2][3] + viewProj[2][0],
        viewProj[3][3] + viewProj[3][0]
      );

      // Right plane: row 4 - row 1
      f.planes[1] = Vector4(
        viewProj[0][3] - viewProj[0][0],
        viewProj[1][3] - viewProj[1][0],
        viewProj[2][3] - viewProj[2][0],
        viewProj[3][3] - viewProj[3][0]
      );

      // Bottom plane: row 4 + row 2
      f.planes[2] = Vector4(
        viewProj[0][3] + viewProj[0][1],
        viewProj[1][3] + viewProj[1][1],
        viewProj[2][3] + viewProj[2][1],
        viewProj[3][3] + viewProj[3][1]
      );

      // Top plane: row 4 - row 2
      f.planes[3] = Vector4(
        viewProj[0][3] - viewProj[0][1],
        viewProj[1][3] - viewProj[1][1],
        viewProj[2][3] - viewProj[2][1],
        viewProj[3][3] - viewProj[3][1]
      );

      // Near plane: row 3 (D3D convention with 0-1 depth range)
      f.planes[4] = Vector4(
        viewProj[0][2],
        viewProj[1][2],
        viewProj[2][2],
        viewProj[3][2]
      );

      // Far plane: row 4 - row 3
      f.planes[5] = Vector4(
        viewProj[0][3] - viewProj[0][2],
        viewProj[1][3] - viewProj[1][2],
        viewProj[2][3] - viewProj[2][2],
        viewProj[3][3] - viewProj[3][2]
      );

      // Normalize all planes
      for (int i = 0; i < 6; i++) {
        float len = std::sqrt(
          f.planes[i].x * f.planes[i].x +
          f.planes[i].y * f.planes[i].y +
          f.planes[i].z * f.planes[i].z
        );
        if (len > 0.0001f) {
          f.planes[i].x /= len;
          f.planes[i].y /= len;
          f.planes[i].z /= len;
          f.planes[i].w /= len;
        }
      }

      return f;
    }

    /**
     * Check if a point is inside the frustum.
     * Returns true if point is on positive side of all planes.
     */
    bool containsPoint(const Vector3& point) const {
      for (int i = 0; i < 6; i++) {
        float dist = planes[i].x * point.x +
                     planes[i].y * point.y +
                     planes[i].z * point.z +
                     planes[i].w;
        if (dist < 0) {
          return false;  // Point is behind this plane
        }
      }
      return true;
    }

    /**
     * Check if an AABB intersects the frustum.
     * Uses conservative test - may return true for boxes that don't actually intersect.
     * Returns false only if box is definitely outside.
     */
    bool intersectsAABB(const AABB& aabb) const {
      if (!aabb.isValid()) {
        return false;
      }

      for (int i = 0; i < 6; i++) {
        // Find the corner of the AABB that is furthest in the direction of the plane normal
        Vector3 pVertex;
        pVertex.x = (planes[i].x >= 0) ? aabb.maxBounds.x : aabb.minBounds.x;
        pVertex.y = (planes[i].y >= 0) ? aabb.maxBounds.y : aabb.minBounds.y;
        pVertex.z = (planes[i].z >= 0) ? aabb.maxBounds.z : aabb.minBounds.z;

        // If this furthest vertex is behind the plane, the box is outside
        float dist = planes[i].x * pVertex.x +
                     planes[i].y * pVertex.y +
                     planes[i].z * pVertex.z +
                     planes[i].w;

        if (dist < 0) {
          return false;  // Box is completely outside this plane
        }
      }

      return true;  // Box is inside or intersecting
    }
  };

  /**
   * Culling information for a draw call
   */
  struct CullingInfo {
    AABB worldSpaceBounds;        // Computed from actual vertex positions
    Matrix4 viewProjection;       // From game's camera setup
    Frustum frustum;              // Derived from viewProjection

    bool boundsAreAccurate;       // We computed bounds, not guessed
    bool disableRemixCulling;     // Force Remix to accept this draw
    bool useGameCullingOnly;      // Trust game's culling decision

    CullingInfo()
      : boundsAreAccurate(false)
      , disableRemixCulling(false)
      , useGameCullingOnly(true) {}
  };

  /**
   * Captured UV data for a geometry
   */
  struct CapturedUVData {
    std::vector<Vector2> uvs;         // Per-vertex UVs
    UVBehavior behavior;              // Classification
    Vector2 scrollRate;               // For animated UVs (units per second)
    float scrollPhase;                // Current phase for animated UVs
    bool isComplete;                  // All UVs captured successfully

    CapturedUVData()
      : behavior(UVBehavior::Unknown)
      , scrollRate(0.0f, 0.0f)
      , scrollPhase(0.0f)
      , isComplete(false) {}
  };

  /**
   * Heavy data structure - stored in cache, looked up by StableID.
   * Contains all processed information about a geometry.
   */
  struct GeometryRecord {
    // Identity
    XXH64_hash_t stableId;

    // Stability tracking (first N frames only)
    uint32_t framesObserved;
    uint32_t hashVariations;
    XXH64_hash_t lastVertexDataHash;
    bool stabilityCheckComplete;
    bool isStable;

    // Classification
    GeometryType geometryType;

    // World-space bounds (updated each frame if geometry moves)
    AABB worldSpaceBounds;
    bool boundsValid;

    // Captured UV data
    CapturedUVData capturedUVs;

    // Texture handling
    TextureType textureType;
    XXH64_hash_t originalTextureHash;
    XXH64_hash_t bakedTextureHash;
    bool needsTextureBake;
    bool textureBakeComplete;

    // Replacement tracking
    bool hasReplacementGeometry;
    bool hasReplacementTexture;
    XXH64_hash_t replacementGeometryHash;
    XXH64_hash_t replacementTextureHash;

    // Culling
    bool hasCullingIssues;
    bool forceFrustumCullOnly;

    // Processing state
    bool isFullyProcessed;
    uint32_t lastSeenFrame;

    GeometryRecord()
      : stableId(0)
      , framesObserved(0)
      , hashVariations(0)
      , lastVertexDataHash(0)
      , stabilityCheckComplete(false)
      , isStable(false)
      , geometryType(GeometryType::Unknown)
      , boundsValid(false)
      , textureType(TextureType::Unknown)
      , originalTextureHash(0)
      , bakedTextureHash(0)
      , needsTextureBake(false)
      , textureBakeComplete(false)
      , hasReplacementGeometry(false)
      , hasReplacementTexture(false)
      , replacementGeometryHash(0)
      , replacementTextureHash(0)
      , hasCullingIssues(false)
      , forceFrustumCullOnly(false)
      , isFullyProcessed(false)
      , lastSeenFrame(0) {}

    /**
     * Update stability tracking with a new vertex data hash
     * Called only during the first N frames of observation
     */
    void updateStabilityTracking(XXH64_hash_t currentVertexHash, uint32_t currentFrame) {
      if (stabilityCheckComplete) return;

      lastSeenFrame = currentFrame;

      if (framesObserved == 0) {
        // First observation
        lastVertexDataHash = currentVertexHash;
      } else if (currentVertexHash != lastVertexDataHash) {
        hashVariations++;
        lastVertexDataHash = currentVertexHash;
      }

      framesObserved++;

      if (framesObserved >= kStabilityCheckFrames) {
        stabilityCheckComplete = true;
        isStable = (hashVariations <= kMaxStableVariations);

        // Classify geometry type based on stability
        if (isStable) {
          geometryType = GeometryType::Static;
        } else if (hashVariations <= 4) {
          geometryType = GeometryType::Skinned;  // Likely bone animation
        } else {
          geometryType = GeometryType::Dynamic;
        }
      }
    }

    /**
     * Check if this geometry needs any processing
     */
    bool needsProcessing() const {
      return !isFullyProcessed ||
             (needsTextureBake && !textureBakeComplete) ||
             (capturedUVs.behavior == UVBehavior::Unknown);
    }
  };

  /**
   * Hash functor for GeometryIdentityKey (for use in std::unordered_map)
   */
  struct GeometryIdentityKeyHash {
    size_t operator()(const GeometryIdentityKey& key) const {
      return static_cast<size_t>(key.computeStableId());
    }
  };

  /**
   * Submission data for Remix integration
   */
  struct GeometrySubmission {
    XXH64_hash_t stableId;

    // Vertex and index data pointers
    const void* vertexData;
    uint32_t vertexCount;
    uint32_t vertexStride;

    const void* indexData;
    uint32_t indexCount;
    uint32_t indexStride;

    // UVs (may be captured/corrected)
    const Vector2* uvData;
    uint32_t uvCount;
    bool uvsAreOverridden;

    // Texture
    XXH64_hash_t textureHash;      // May be original or baked hash
    bool textureIsReplacement;

    // Bounds and culling
    AABB worldSpaceBounds;
    CullingInfo cullingInfo;

    // Transforms
    Matrix4 objectToWorld;

    // Flags
    bool skipRemixProcessing;      // Don't let Remix process this
    bool forceVisible;             // Override visibility checks
    bool hasReplacementAsset;      // Has a replacement geometry/texture

    GeometrySubmission()
      : stableId(0)
      , vertexData(nullptr)
      , vertexCount(0)
      , vertexStride(0)
      , indexData(nullptr)
      , indexCount(0)
      , indexStride(0)
      , uvData(nullptr)
      , uvCount(0)
      , uvsAreOverridden(false)
      , textureHash(0)
      , textureIsReplacement(false)
      , skipRemixProcessing(false)
      , forceVisible(false)
      , hasReplacementAsset(false) {}
  };

} // namespace dxvk
