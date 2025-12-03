# Shader Compatibility System - Implementation Plan

## Overview

A comprehensive compatibility layer to make shader-based D3D9 games work correctly with RTX Remix (designed for fixed-function games).

## Core Problems to Solve

| Problem | Impact | Priority |
|---------|--------|----------|
| Unstable geometry hashes | Geometry not recognized frame-to-frame | P0 |
| Culling conflicts | Geometry disappears incorrectly | P0 |
| Sliding/distorted UVs | Textures don't align correctly | P1 |
| Render target textures | Dynamic textures not captured | P1 |
| Procedural textures | No source texture to use | P2 |
| VRAM efficiency | Double texture storage wastes memory | P0 |

---

# PHASE 1: Stable Geometry Identification

## Problem Analysis

Remix identifies geometry by hashing vertex data. This fails when:
1. **Skinned meshes** - Vertices change every frame due to bone transforms
2. **Morphing geometry** - Blend shapes modify vertices
3. **Procedural geometry** - Vertices computed in shader
4. **Dynamic buffers** - Game reuses buffers with different data

## Solution: Multi-Factor Stable ID

Instead of hashing vertex DATA, hash IDENTITY factors that remain constant:

```
StableID = Hash(
    vertexShaderHash,      // Same VS = same mesh type
    pixelShaderHash,       // Same PS = same material type
    vertexCount,           // Topology indicator
    indexCount,            // Topology indicator
    vertexStride,          // Format indicator
    primitiveTopology,     // Triangle list, strip, etc.
    textureBindingHashes[], // What textures are bound
    renderTargetHash,      // Where it's rendering to
    drawCallOrderInFrame   // Disambiguate similar draws
)
```

## Implementation Design

### Data Structures

```cpp
// Lightweight - computed every frame, must be fast
struct GeometryIdentityKey {
    XXH64_hash_t vsHash;
    XXH64_hash_t psHash;
    uint32_t vertexCount;
    uint32_t indexCount;
    uint32_t vertexStride;
    uint32_t primitiveTopology;
    XXH64_hash_t textureBindingsHash;  // Combined hash of bound textures
    uint32_t drawOrderIndex;           // Nth draw call this frame

    XXH64_hash_t computeStableId() const;
};

// Heavy data - stored in cache, looked up by StableID
struct GeometryRecord {
    XXH64_hash_t stableId;

    // Stability tracking
    uint32_t framesObserved;
    uint32_t hashVariations;           // How many different vertex hashes seen
    bool isStable;                     // Vertex hash same across frames?

    // Cached vertex hash (for stable geometry)
    XXH64_hash_t lastVertexDataHash;

    // Replacement data
    bool hasReplacement;
    XXH64_hash_t replacementAssetHash;

    // Flags
    bool needsUVCapture;
    bool needsTextureBake;
    bool hasCullingIssues;
};
```

### Algorithm

```
PER FRAME:
    drawOrderCounter = 0

    FOR EACH draw call:
        drawOrderCounter++

        // Step 1: Compute identity key (FAST - no vertex data access)
        key = GeometryIdentityKey {
            vsHash = currentVertexShader.hash,
            psHash = currentPixelShader.hash,
            vertexCount = drawParams.vertexCount,
            indexCount = drawParams.indexCount,
            vertexStride = vertexBuffer.stride,
            primitiveTopology = currentTopology,
            textureBindingsHash = hashBoundTextures(),
            drawOrderIndex = drawOrderCounter
        }

        stableId = key.computeStableId()

        // Step 2: Lookup or create record
        record = geometryCache.getOrCreate(stableId)

        // Step 3: Track stability (first N frames only)
        IF record.framesObserved < STABILITY_CHECK_FRAMES:
            currentVertexHash = hashVertexData()  // Expensive, only during detection
            IF currentVertexHash != record.lastVertexDataHash:
                record.hashVariations++
            record.lastVertexDataHash = currentVertexHash
            record.framesObserved++

            IF record.framesObserved == STABILITY_CHECK_FRAMES:
                record.isStable = (record.hashVariations <= 1)

        // Step 4: Use stable ID for Remix submission
        submitToRemix(stableId, record, drawCall)
```

### Performance Considerations

1. **Identity key computation**: O(1) - just reading existing state, no vertex access
2. **Stable ID hash**: Single XXH64 call on ~48 bytes - negligible
3. **Cache lookup**: O(1) hash map lookup
4. **Vertex hashing**: Only during detection phase (first N frames per geometry)
5. **Memory**: ~128 bytes per unique geometry record

### Hash Function Selection

Use **XXH3** (not XXH64) for stable ID computation:
- 2-3x faster than XXH64 for small inputs
- Better distribution for small keys
- Available in xxHash library already in codebase

```cpp
XXH64_hash_t GeometryIdentityKey::computeStableId() const {
    // Pack into contiguous memory for single hash call
    struct alignas(8) PackedKey {
        uint64_t vsHash;
        uint64_t psHash;
        uint32_t vertexCount;
        uint32_t indexCount;
        uint32_t vertexStride;
        uint32_t primitiveTopology;
        uint64_t textureBindingsHash;
        uint32_t drawOrderIndex;
        uint32_t padding;
    };

    PackedKey packed = {
        vsHash, psHash, vertexCount, indexCount,
        vertexStride, primitiveTopology, textureBindingsHash,
        drawOrderIndex, 0
    };

    return XXH3_64bits(&packed, sizeof(packed));
}
```

### Edge Cases

| Case | Detection | Handling |
|------|-----------|----------|
| Same mesh, different textures | Different textureBindingsHash | Separate stable IDs |
| Same mesh, different shaders | Different vsHash/psHash | Separate stable IDs |
| Instanced draws | Same everything, different drawOrder | drawOrderIndex disambiguates |
| Multi-pass rendering | Same geometry, different RT | renderTargetHash could be added |
| LOD switches | Different vertexCount | Separate stable IDs (correct) |

---

# PHASE 2: Culling Fix

## Problem Analysis

The game performs its own culling (CPU-side, before draw calls). Remix may:
1. Apply additional culling that conflicts
2. Miss geometry because game's culling uses data Remix doesn't see
3. Cull based on bounding boxes that are wrong for transformed geometry

## Solution: Culling Passthrough with Bounds Override

### Approach

1. **Trust game's culling** - If game submits a draw call, it passed the game's culling
2. **Provide accurate bounds** - Give Remix correct world-space bounding boxes
3. **Fix frustum data** - Ensure Remix uses correct camera frustum

### Implementation

```cpp
struct CullingInfo {
    // Computed from actual vertex positions (world space)
    AABB worldSpaceBounds;

    // From game's camera setup
    Matrix4 viewProjection;
    Frustum frustum;

    // Override flags
    bool boundsAreAccurate;      // We computed bounds, not guessed
    bool disableRemixCulling;    // Force Remix to accept this draw
};

// Compute accurate bounds from transformed vertices
AABB computeWorldSpaceBounds(
    const DrawCallState& draw,
    const Matrix4& objectToWorld
) {
    AABB bounds = AABB::empty();

    // Sample vertices (don't need all, just enough for bounds)
    const uint32_t sampleCount = min(draw.vertexCount, 64u);
    const uint32_t stride = max(1u, draw.vertexCount / sampleCount);

    for (uint32_t i = 0; i < draw.vertexCount; i += stride) {
        Vector3 localPos = readVertexPosition(draw, i);
        Vector3 worldPos = objectToWorld.transformPoint(localPos);
        bounds.expand(worldPos);
    }

    return bounds;
}
```

### Integration Points

1. **Where bounds are computed**: After vertex shader constants captured, before Remix submission
2. **Where frustum is set**: From game's view/projection matrices (already captured)
3. **Where override happens**: In Remix's geometry submission path

---

# PHASE 3: UV Capture System

## Problem Analysis

UVs slide/distort because:
1. Game's VS computes UVs using matrices/constants we're not applying
2. We're using raw UV data instead of transformed UV output
3. View-dependent UV calculations produce different results each frame

## Solution: Capture VS Output UVs

### Approach Options

| Option | Performance | Complexity | Accuracy |
|--------|-------------|------------|----------|
| A. CPU VS emulation | Medium | High | Perfect |
| B. Transform feedback | Fast | Medium | Perfect |
| C. Compute shader replication | Fast | High | Perfect |
| D. Sample at reference view | Fast | Low | Good enough |

**Recommended: Option D for static, Option A for complex cases**

### Option D: Reference View Sampling (Primary)

For most geometry, UVs are static or only depend on object-space position:

```cpp
struct UVCaptureResult {
    std::vector<Vector2> uvs;        // Per-vertex UVs
    XXH64_hash_t geometryStableId;
    bool isComplete;
};

// Capture UVs once when geometry first seen
UVCaptureResult captureUVsAtReferenceView(const DrawCallState& draw) {
    UVCaptureResult result;
    result.geometryStableId = draw.stableId;

    // Read UV data directly from vertex buffer
    // This is the UV BEFORE vertex shader transformation
    for (uint32_t i = 0; i < draw.vertexCount; i++) {
        Vector2 uv = readVertexUV(draw, i);
        result.uvs.push_back(uv);
    }

    result.isComplete = true;
    return result;
}
```

### Option A: CPU VS Emulation (Fallback for complex shaders)

For shaders that compute UVs procedurally:

```cpp
// Simplified VS interpreter for UV computation only
class VertexShaderUVEmulator {
    std::array<Vector4, 256> constants;  // c0-c255

    Vector2 computeOutputUV(
        const Vector3& position,
        const Vector2& inputUV,
        const DxsoProgram& vsProgram
    ) {
        // Only execute instructions that affect UV output
        // Skip position transformation, lighting, etc.
        // ...
    }
};
```

### UV Classification

```cpp
enum class UVBehavior {
    STATIC,              // UV from vertex buffer, no transformation
    TRANSFORMED,         // UV modified by VS but result is static
    ANIMATED,            // UV changes over time (scrolling)
    VIEW_DEPENDENT,      // UV depends on camera (rare, needs special handling)
    PROCEDURAL           // UV computed entirely in shader
};

UVBehavior classifyUVBehavior(const GeometryRecord& record) {
    // Check if VS modifies texture coordinates
    if (!record.vsModifiesUV) return UVBehavior::STATIC;

    // Check if UV output is stable across frames
    if (record.uvHashStable) return UVBehavior::TRANSFORMED;

    // Check for time-based constants
    if (record.usesTimeConstant) return UVBehavior::ANIMATED;

    // Check for view matrix usage in UV calculation
    if (record.uvUsesViewMatrix) return UVBehavior::VIEW_DEPENDENT;

    return UVBehavior::PROCEDURAL;
}
```

---

# PHASE 4: Texture Handling (VRAM Efficient)

## Problem: Double Texture Storage

Naive approach stores:
- Original texture (game's)
- Baked texture (our capture)
= 2x VRAM usage

## Solution: Texture Replacement, Not Duplication

### Strategy

```
IF texture needs baking:
    1. Bake to NEW texture
    2. Register baked texture hash with Remix
    3. RELEASE original texture reference (if possible)
    4. Or: SHARE memory via aliasing

IF texture is passthrough (no baking needed):
    1. Use original texture directly
    2. No duplication
```

### Implementation

```cpp
class TextureManager {
    // Map from original texture hash to our handling
    std::unordered_map<XXH64_hash_t, TextureRecord> m_records;

    struct TextureRecord {
        TextureType type;              // STATIC, RENDER_TARGET, PROCEDURAL

        // Original texture (may be released after baking)
        Rc<DxvkImage> originalTexture;
        bool originalReleased;

        // Baked texture (only if needed)
        Rc<DxvkImage> bakedTexture;

        // Memory optimization
        bool usesMemoryAliasing;       // Baked reuses original's memory
    };

    Rc<DxvkImage> getTextureForSubmission(XXH64_hash_t originalHash) {
        auto& record = m_records[originalHash];

        switch (record.type) {
            case TextureType::STATIC:
                // No baking needed, use original
                return record.originalTexture;

            case TextureType::RENDER_TARGET:
            case TextureType::PROCEDURAL:
                // Use baked version
                return record.bakedTexture;
        }
    }
};
```

### Memory Aliasing for Baked Textures

When baking, reuse the original texture's memory if possible:

```cpp
Rc<DxvkImage> bakeTextureInPlace(
    Rc<DxvkImage> original,
    const DrawCallState& draw
) {
    // If format and size match, we can alias memory
    if (canAliasMemory(original, requiredFormat, requiredSize)) {
        // Create view into same memory with different format
        // Render baked content
        // Original and baked share memory (1x VRAM)
        return createAliasedImage(original, requiredFormat);
    }

    // Otherwise allocate new (2x VRAM temporarily)
    auto baked = allocateTexture(requiredFormat, requiredSize);
    renderBakedContent(baked, draw);

    // Release original to reclaim memory
    original = nullptr;

    return baked;
}
```

### Render Target Textures

For textures that are render targets (game renders to them):

```cpp
enum class RTCaptureStrategy {
    CAPTURE_ONCE,           // RT content is static, capture once
    CAPTURE_ON_CHANGE,      // RT updates sometimes, capture when dirty
    CAPTURE_EVERY_FRAME,    // RT updates every frame (expensive, avoid)
    PASSTHROUGH             // Let Remix handle RT directly
};

RTCaptureStrategy classifyRenderTarget(const RenderTargetInfo& rt) {
    // Track RT writes over multiple frames
    if (rt.writesPerFrame == 0) return RTCaptureStrategy::PASSTHROUGH;
    if (rt.contentHashStable) return RTCaptureStrategy::CAPTURE_ONCE;
    if (rt.writesPerFrame < 0.1f) return RTCaptureStrategy::CAPTURE_ON_CHANGE;
    return RTCaptureStrategy::CAPTURE_EVERY_FRAME;  // Last resort
}
```

### Procedural Texture Baking

For textures generated entirely by pixel shader:

```cpp
struct ProceduralBakeRequest {
    XXH64_hash_t psHash;           // Pixel shader that generates texture
    ShaderConstants constants;      // Constants used
    Vector2u resolution;            // Output resolution
    VkFormat format;               // Output format

    bool isTimeDependant;          // Uses time constant?
    bool isViewDependant;          // Uses view matrix?
};

Rc<DxvkImage> bakeProceduralTexture(const ProceduralBakeRequest& req) {
    // Render fullscreen quad with the pixel shader
    // Capture output to texture
    // ...
}
```

---

# PHASE 5: Integration & Submission

## Remix Submission Pipeline

```cpp
void ShaderCompatibilityManager::processAndSubmit(
    RtxContext* ctx,
    const DrawCallState& draw
) {
    // 1. Get or create stable ID
    auto stableId = computeStableId(draw);
    auto& record = m_geometryCache.getOrCreate(stableId);

    // 2. Check if fully processed
    if (!record.isFullyProcessed) {
        processGeometry(ctx, draw, record);
    }

    // 3. Prepare submission data
    GeometrySubmission submission;
    submission.stableId = stableId;
    submission.vertexData = getVertexData(draw, record);
    submission.indexData = getIndexData(draw);
    submission.uvData = getUVData(draw, record);
    submission.texture = getTexture(draw, record);
    submission.worldSpaceBounds = computeBounds(draw, record);

    // 4. Submit to Remix
    ctx->submitGeometryForRayTracing(submission);
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRAME N                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Draw Call ──┬──▶ Compute StableID ──▶ Cache Lookup             │
│              │           │                   │                  │
│              │           ▼                   ▼                  │
│              │    ┌─────────────┐    ┌─────────────┐           │
│              │    │ First time? │───▶│   Process   │           │
│              │    └─────────────┘    │  - Classify │           │
│              │           │           │  - Capture UV│           │
│              │           │ No        │  - Bake tex │           │
│              │           ▼           └─────────────┘           │
│              │    ┌─────────────┐           │                  │
│              │    │ Use cached  │◀──────────┘                  │
│              │    └─────────────┘                              │
│              │           │                                      │
│              ▼           ▼                                      │
│         Compute     Prepare Submission                          │
│         Bounds      (StableID, UVs, Texture)                   │
│              │           │                                      │
│              └─────┬─────┘                                      │
│                    ▼                                            │
│            Submit to Remix                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Implementation Order

## Phase 1: Foundation (Week 1)
- [ ] GeometryIdentityKey structure
- [ ] StableID computation (XXH3)
- [ ] GeometryRecord cache
- [ ] Stability detection (multi-frame tracking)
- [ ] Integration point in draw call path

## Phase 2: Culling (Week 1)
- [ ] World-space bounds computation
- [ ] Bounds sampling algorithm
- [ ] Frustum data passthrough
- [ ] Remix culling override mechanism

## Phase 3: UV System (Week 2)
- [ ] UV behavior classification
- [ ] Static UV capture
- [ ] UV storage in GeometryRecord
- [ ] Animated UV parameter extraction

## Phase 4: Texture System (Week 2)
- [ ] TextureManager class
- [ ] Texture type classification
- [ ] Memory-efficient baking
- [ ] RT capture strategies

## Phase 5: Integration (Week 3)
- [ ] Submission pipeline
- [ ] Remix API integration
- [ ] Replacement asset loading
- [ ] Debug visualization

## Phase 6: Testing & Edge Cases (Week 3)
- [ ] Stability testing
- [ ] Performance profiling
- [ ] Edge case handling
- [ ] Memory usage validation

---

# Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| StableID computation | < 100ns | No vertex data access |
| Cache lookup | < 50ns | Hash map O(1) |
| Full processing (first time) | < 1ms | One-time per geometry |
| Per-frame submission | < 10μs | Just lookups and submission |
| Memory per geometry | < 256 bytes | Excluding UV/texture data |
| VRAM overhead | < 10% | No unnecessary duplication |

---

# Files to Create/Modify

## New Files
- `src/dxvk/rtx_render/rtx_shader_compat_manager.h`
- `src/dxvk/rtx_render/rtx_shader_compat_manager.cpp`
- `src/dxvk/rtx_render/rtx_geometry_identity.h`
- `src/dxvk/rtx_render/rtx_texture_manager.h`
- `src/dxvk/rtx_render/rtx_uv_capture.h`

## Modified Files
- `src/dxvk/rtx_render/rtx_context.cpp` - Integration point
- `src/d3d9/d3d9_rtx.cpp` - Draw call interception
- `src/dxvk/rtx_render/rtx_materials.cpp` - Texture handling

---

# Risk Mitigation

| Risk | Mitigation |
|------|------------|
| StableID collisions | Add more factors (RT hash, blend state) if needed |
| UV capture misses edge cases | Fallback to per-frame computation for flagged geometry |
| VRAM still too high | Implement texture streaming / compression |
| Performance regression | Profile continuously, optimize hot paths |
| Remix API limitations | Work with Remix team or use workarounds |
