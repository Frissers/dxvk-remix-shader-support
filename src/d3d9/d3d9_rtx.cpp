#pragma once

#include "d3d9_include.h"
#include "d3d9_state.h"

#include "d3d9_util.h"
#include "d3d9_buffer.h"

#include "d3d9_rtx.h"
#include "d3d9_device.h"

#include "../util/util_fastops.h"
#include "../util/util_math.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_texture.h"
#include "d3d9_surface.h"
#include "../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../dxvk/rtx_render/rtx_shader_output_capturer.h"
#include "../dxvk/rtx_render/rtx_shader_compatibility_manager.h"
#include "../util/xxHash/xxhash.h"
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace dxvk {
  // Global matrix cache from SetVertexShaderConstantF
  // Stores matrices by their starting register number (e.g., c4 -> index 4)
  // Used by processRenderState to get view/proj matrices based on shader analysis
  std::array<Matrix4, 256> g_cachedMatricesByRegister;
  std::array<bool, 256> g_hasMatrixAtRegister = {};  // Initialize all to false

  // Transform cache from SetTransform (D3DTS_VIEW, D3DTS_PROJECTION)
  // Games using vertex shaders often still use SetTransform for camera matrices
  Matrix4 g_cachedViewTransform;
  Matrix4 g_cachedProjectionTransform;
  bool g_hasViewTransform = false;
  bool g_hasProjectionTransform = false;

  // Track the view matrix register identified by shader analysis
  // This is set by DrawIndexedPrimitive when it finds a shader with view matrix info
  // processRenderState will check this register FIRST before scanning others
  int g_shaderAnalyzedViewReg = -1;

  // Camera position from c8 (vs_worldCamPos) - some games store camera position separately
  // from the view matrix. We cache it here to construct proper view matrices.
  Vector4 g_cachedCameraPosition = Vector4(0, 0, 0, 1);
  bool g_hasCachedCameraPosition = false;

  // Forward declaration for global shader registry (defined in dxvk_imgui.cpp)
  struct VertexShaderInfo {
    uint64_t hash;
    uint32_t drawCallCount;
    std::string name;
  };

  // Extern declarations for global shader registry
  extern std::unordered_map<uint64_t, VertexShaderInfo> g_vertexShaderRegistry;
  extern std::mutex g_vertexShaderRegistryMutex;
  
  static const bool s_isDxvkResolutionEnvVarSet = (env::getEnvVar("DXVK_RESOLUTION_WIDTH") != "") || (env::getEnvVar("DXVK_RESOLUTION_HEIGHT") != "");
  
  // We only look at RT 0 currently.
  const uint32_t kRenderTargetIndex = 0;

  #define CATEGORIES_REQUIRE_DRAW_CALL_STATE  InstanceCategories::Sky, InstanceCategories::Terrain
  #define CATEGORIES_REQUIRE_GEOMETRY_COPY    InstanceCategories::Terrain, InstanceCategories::WorldUI, InstanceCategories::LegacyEmissive

  D3D9Rtx::D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion)
    : m_rtStagingData(d3d9Device->GetDXVKDevice(), "RtxStagingDataAlloc: D3D9", (VkMemoryPropertyFlagBits) (VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
    , m_parent(d3d9Device)
    , m_enableDrawCallConversion(enableDrawCallConversion)
    , m_pGeometryWorkers(enableDrawCallConversion ? std::make_unique<GeometryProcessor>(numGeometryProcessingThreads(), "geometry-processing") : nullptr)
    , m_shaderCompatibilityManager(std::make_unique<ShaderCompatibilityManager>()) {

    // Add space for 256 objects skinned with 256 bones each.
    m_stagedBones.resize(256 * 256);
  }

  D3D9Rtx::~D3D9Rtx() {
    // ShaderCompatibilityManager destructor handles cleanup
  }

  void D3D9Rtx::Initialize() {
    m_vsVertexCaptureData = m_parent->CreateConstantBuffer(false,
                                        sizeof(D3D9RtxVertexCaptureData),
                                        DxsoProgramType::VertexShader,
                                        DxsoConstantBuffers::VSVertexCaptureData);

    // Get constant buffer bindings from D3D9
    m_parent->EmitCs([vertexCaptureCB = m_vsVertexCaptureData](DxvkContext* ctx) {
      const uint32_t vsFixedFunctionConstants = computeResourceSlotId(DxsoProgramType::VertexShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::VSFixedFunction);
      const uint32_t psSharedStateConstants = computeResourceSlotId(DxsoProgramType::PixelShader, DxsoBindingType::ConstantBuffer, DxsoConstantBuffers::PSShared);
      static_cast<RtxContext*>(ctx)->setConstantBuffers(vsFixedFunctionConstants, psSharedStateConstants, vertexCaptureCB);
    });
  }

  const Direct3DState9& D3D9Rtx::d3d9State() const {
    return *m_parent->GetRawState();
  }

  template<typename T>
  void D3D9Rtx::copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    assert(indexCount >= 3);

    // Find min/max index
    {
      ScopedCpuProfileZoneN("Find min/max");

      fast::findMinMax<T>(indexCount, pIndices, minIndex, maxIndex);
    }

    // Modify the indices if the min index is non-zero
    {
      ScopedCpuProfileZoneN("Copy indices");

      if (minIndex != 0) {
        fast::copySubtract<T>(pIndicesDst, pIndices, indexCount, (T) minIndex);
      } else {
        memcpy(pIndicesDst, pIndices, sizeof(T) * indexCount);
      }
    }
  }

  template<typename T>
  DxvkBufferSlice D3D9Rtx::processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex) {
    ScopedCpuProfileZone();

    const uint32_t indexStride = sizeof(T);
    const size_t numIndexBytes = indexCount * indexStride;
    const size_t indexOffset = indexStride * startIndex;

    auto processing = [this, &indexCtx, indexCount](const size_t offset, const size_t size) -> D3D9CommonBuffer::RemixIndexBufferMemoizationData {
      D3D9CommonBuffer::RemixIndexBufferMemoizationData result;

      // Get our slice of the staging ring buffer
      result.slice = m_rtStagingData.alloc(CACHE_LINE_SIZE, size);

      // Acquire prevents the staging allocator from re-using this memory
      result.slice.buffer()->acquire(DxvkAccess::Read);

      const uint8_t* pBaseIndex = (uint8_t*) indexCtx.indexBuffer.mapPtr + offset;

      T* pIndices = (T*) pBaseIndex;
      T* pIndicesDst = (T*) result.slice.mapPtr(0);
      copyIndices<T>(indexCount, pIndicesDst, pIndices, result.min, result.max);

      return result;
    };

    if (enableIndexBufferMemoization() && indexCtx.ibo != nullptr) {
      // If we have an index buffer, we can utilize memoization
      D3D9CommonBuffer::RemixIboMemoizer& memoization = indexCtx.ibo->remixMemoization;
      const auto result = memoization.memoize(indexOffset, numIndexBytes, processing);
      minIndex = result.min;
      maxIndex = result.max;
      return result.slice;
    }

    // No index buffer (so no memoization) - this could be a DrawPrimitiveUP call (where IB data is passed inline)
    const auto result = processing(indexOffset, numIndexBytes);
    minIndex = result.min;
    maxIndex = result.max;
    return result.slice;
  }

  DxvkBufferSlice allocVertexCaptureBuffer(DxvkDevice* pDevice, const VkDeviceSize size) {
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.access = VK_ACCESS_TRANSFER_READ_BIT;
    info.stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.size = size;
    return DxvkBufferSlice(pDevice->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::AppBuffer, "Vertex Capture Buffer"));
  }

  void D3D9Rtx::prepareVertexCapture(const int vertexIndexOffset) {
    ScopedCpuProfileZone();

    static_assert(sizeof CapturedVertex == 80, "The injected shader code is expecting this exact structure size to work correctly, see emitVertexCaptureWrite in dxso_compiler.cpp");

    auto BoundShaderHas = [&](const D3D9CommonShader* shader, DxsoUsage usage, uint32_t usageIndex, bool inOut)-> bool {
      if (shader == nullptr)
        return false;

      const auto& sgn = inOut ? shader->GetIsgn() : shader->GetOsgn();
      for (uint32_t i = 0; i < sgn.elemCount; i++) {
        const auto& decl = sgn.elems[i];
        if (decl.semantic.usageIndex == usageIndex && decl.semantic.usage == usage)
          return true;
      }
      return false;
    };

    // Get common shaders to query what data we can capture
    const D3D9CommonShader* vertexShader = d3d9State().vertexShader.ptr() != nullptr ? d3d9State().vertexShader->GetCommonShader() : nullptr;

    // Track vertex shader usage and check if we should capture from this shader
    uint64_t currentShaderHash = 0;
    bool shouldCapture = true;
    
    if (vertexShader != nullptr && d3d9State().vertexShader.ptr() != nullptr) {
      // Get shader bytecode for hashing
      auto& shaderByteCode = vertexShader->GetBytecode();
      currentShaderHash = XXH64(shaderByteCode.data(), shaderByteCode.size(), 0);
      
      // Update global shader registry for ImGui
      {
        std::lock_guard<std::mutex> lock(g_vertexShaderRegistryMutex);
        auto& info = g_vertexShaderRegistry[currentShaderHash];
        info.hash = currentShaderHash;
        info.drawCallCount++;
        if (info.name.empty()) {
          info.name = str::format("VS_", std::hex, currentShaderHash);
        }
      }
      
      // Check if we should only capture from specific shaders
      const auto& targetHashes = vertexShadersToCapture();
      if (!targetHashes.empty() && targetHashes.count(currentShaderHash) == 0) {
        shouldCapture = false;
      }
    }
    
    // If this shader isn't selected for capture, return early
    // This leaves the geometry buffers as set by processVertices (fixed function path)
    if (!shouldCapture) {
      return;
    }

    RasterGeometry& geoData = m_activeDrawCallState.geometryData;

    // Known stride for vertex capture buffers
    const uint32_t stride = sizeof(CapturedVertex);
    const size_t vertexCaptureDataSize = align(geoData.vertexCount * stride, CACHE_LINE_SIZE);

    DxvkBufferSlice slice = allocVertexCaptureBuffer(m_parent->GetDXVKDevice().ptr(), vertexCaptureDataSize);

    geoData.positionBuffer = RasterBuffer(slice, 0, stride, VK_FORMAT_R32G32B32A32_SFLOAT);
    assert(geoData.positionBuffer.offset() % 4 == 0);

    // Capture TEXCOORD0
    if (BoundShaderHas(vertexShader, DxsoUsage::Texcoord, 0, false) && (!geoData.texcoordBuffer.defined() || !RtxGeometryUtils::isTexcoordFormatValid(geoData.texcoordBuffer.vertexFormat()))) {
      const uint32_t texcoordOffset = offsetof(CapturedVertex, texcoord0);
      geoData.texcoordBuffer = RasterBuffer(slice, texcoordOffset, stride, VK_FORMAT_R32G32_SFLOAT);
      assert(geoData.texcoordBuffer.offset() % 4 == 0);
    }

    // Capture TEXCOORD1 (for lightmaps, detail textures)
    if (useVertexCapturedTexcoords() && BoundShaderHas(vertexShader, DxsoUsage::Texcoord, 1, false)) {
      const uint32_t texcoord1Offset = offsetof(CapturedVertex, texcoord1);
      geoData.texcoord1Buffer = RasterBuffer(slice, texcoord1Offset, stride, VK_FORMAT_R32G32_SFLOAT);
      assert(geoData.texcoord1Buffer.offset() % 4 == 0);
    }

    // Check if we should/can get normals
    if ((BoundShaderHas(vertexShader, DxsoUsage::Normal, 0, false) || BoundShaderHas(vertexShader, DxsoUsage::Normal, 0, true)) && useVertexCapturedNormals()) {
      const uint32_t normalOffset = offsetof(CapturedVertex, normal0);
      geoData.normalBuffer = RasterBuffer(slice, normalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.normalBuffer.offset() % 4 == 0);
    } else {
      geoData.normalBuffer = RasterBuffer();
    }

    // Capture Tangent (for normal mapping)
    if (useVertexCapturedTangents() && BoundShaderHas(vertexShader, DxsoUsage::Tangent, 0, false)) {
      const uint32_t tangentOffset = offsetof(CapturedVertex, tangent0);
      geoData.tangentBuffer = RasterBuffer(slice, tangentOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.tangentBuffer.offset() % 4 == 0);
    }

    // Capture Binormal (for normal mapping)
    if (useVertexCapturedTangents() && BoundShaderHas(vertexShader, DxsoUsage::Binormal, 0, false)) {
      const uint32_t binormalOffset = offsetof(CapturedVertex, binormal0);
      geoData.binormalBuffer = RasterBuffer(slice, binormalOffset, stride, VK_FORMAT_R32G32B32_SFLOAT);
      assert(geoData.binormalBuffer.offset() % 4 == 0);
    }

    // Capture COLOR0
    if (BoundShaderHas(vertexShader, DxsoUsage::Color, 0, false) && d3d9State().pixelShader.ptr() == nullptr) {
      const uint32_t colorOffset = offsetof(CapturedVertex, color0);
      geoData.color0Buffer = RasterBuffer(slice, colorOffset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color0Buffer.offset() % 4 == 0);
    }

    // Capture COLOR1 (specular color)
    if (useVertexCapturedColor1() && BoundShaderHas(vertexShader, DxsoUsage::Color, 1, false)) {
      const uint32_t color1Offset = offsetof(CapturedVertex, color1);
      geoData.color1Buffer = RasterBuffer(slice, color1Offset, stride, VK_FORMAT_B8G8R8A8_UNORM);
      assert(geoData.color1Buffer.offset() % 4 == 0);
    }

    auto constants = m_vsVertexCaptureData->allocSlice();

    // Upload
    auto& data = *reinterpret_cast<D3D9RtxVertexCaptureData*>(constants.mapPtr);

    // ASPECT RATIO FIX FOR SHADER CAPTURE:
    // When shader reexecution is enabled, we need to inverse-transform clip-space positions
    // back to object space. If the game's projection has wrong aspect ratio, the captured
    // geometry will be distorted.
    //
    // The fix: If the projection's aspect ratio is NOT a common widescreen ratio (16:9, 16:10, 21:9, 4:3),
    // correct it to 16:9 since that's the most common modern display ratio.
    Matrix4 correctedProj = m_activeDrawCallState.transformData.viewToProjection;
    {
      float projScaleX = std::abs(correctedProj[0][0]);
      float projScaleY = std::abs(correctedProj[1][1]);
      float projAspect = (projScaleX > 0.001f) ? projScaleY / projScaleX : 1.0f;

      // Common aspect ratios to NOT correct (these are likely intentional):
      // 16:9 = 1.777, 16:10 = 1.6, 21:9 = 2.333, 4:3 = 1.333
      bool isCommonAspect =
        (projAspect > 1.72f && projAspect < 1.83f) ||   // 16:9 range
        (projAspect > 1.55f && projAspect < 1.65f) ||   // 16:10 range
        (projAspect > 2.28f && projAspect < 2.40f) ||   // 21:9 range
        (projAspect > 1.28f && projAspect < 1.38f);     // 4:3 range

      // Only correct if NOT a common aspect (meaning the game has a weird internal aspect)
      // AND only if projAspect is close to 1:1 (which often indicates a bug)
      if (projScaleX > 0.001f && projScaleY > 0.001f && !isCommonAspect) {
        // Correct to 16:9
        float targetAspect = 16.0f / 9.0f;
        float correctedScaleX = projScaleY / targetAspect;
        float sign = (correctedProj[0][0] >= 0) ? 1.0f : -1.0f;
        correctedProj[0][0] = sign * correctedScaleX;

        static uint32_t s_captureAspectFixLogCount = 0;
        if (s_captureAspectFixLogCount++ < 20) {
          Logger::info(str::format("[CAPTURE-ASPECT-FIX] Corrected non-standard projection:",
            " origScaleX=", projScaleX, " scaleY=", projScaleY,
            " projAspect=", projAspect, " correctedScaleX=", correctedScaleX));
        }
      }
    }

    data.invProj = inverse(correctedProj);
    data.viewToWorld = inverseAffine(m_activeDrawCallState.transformData.worldToView);
    data.worldToObject = inverseAffine(m_activeDrawCallState.transformData.objectToWorld);
    data.normalTransform = m_activeDrawCallState.transformData.objectToWorld;
    data.baseVertex = (uint32_t)std::max(0, vertexIndexOffset);

    m_parent->EmitCs([cVertexDataSlice = slice,
                      cConstantBuffer = m_vsVertexCaptureData,
                      cConstants = constants](DxvkContext* ctx) {
      // Bind the new constants to buffer
      ctx->invalidateBuffer(cConstantBuffer, cConstants);

      // Invalidate rest of the members
      // customWorldToProjection is not invalidated as its use is controlled by D3D9SpecConstantId::CustomVertexTransformEnabled being enabled
      ctx->bindResourceBuffer(getVertexCaptureBufferSlot(), cVertexDataSlice);
    });
  }

  void D3D9Rtx::processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData) {
    DxvkBufferSlice streamCopies[caps::MaxStreams] {};

    // Process vertex buffers from CPU
    for (const auto& element : d3d9State().vertexDecl->GetElements()) {
      // Get vertex context
      const VertexContext& ctx = vertexContext[element.Stream];

      if (ctx.mappedSlice.handle == VK_NULL_HANDLE)
        continue;

      ScopedCpuProfileZoneN("Process Vertices");
      const int32_t vertexOffset = ctx.offset + ctx.stride * vertexIndexOffset;
      const uint32_t numVertexBytes = ctx.stride * geoData.vertexCount;

      // Validating index data here, vertexCount and vertexIndexOffset accounts for the min/max indices
      if (RtxOptions::validateCPUIndexData()) {
        if (ctx.mappedSlice.length < vertexOffset + numVertexBytes) {
          throw DxvkError("Invalid draw call");
        }
      }

      // TODO: Simplify this by refactoring RasterGeometry to contain an array of RasterBuffer's
      RasterBuffer* targetBuffer = nullptr;
      switch (element.Usage) {
      case D3DDECLUSAGE_POSITIONT:
      case D3DDECLUSAGE_POSITION:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.positionBuffer;
        break;
      case D3DDECLUSAGE_BLENDWEIGHT:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendWeightBuffer;
        break;
      case D3DDECLUSAGE_BLENDINDICES:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.blendIndicesBuffer;
        break;
      case D3DDECLUSAGE_NORMAL:
        if (element.UsageIndex == 0)
          targetBuffer = &geoData.normalBuffer;
        break;
      case D3DDECLUSAGE_TEXCOORD:
        if (m_texcoordIndex <= MAXD3DDECLUSAGEINDEX && element.UsageIndex == m_texcoordIndex)
          targetBuffer = &geoData.texcoordBuffer;
        break;
      case D3DDECLUSAGE_COLOR:
        if (element.UsageIndex == 0 &&
            !RtxOptions::ignoreAllVertexColorBakedLighting()) {
          const XXH64_hash_t textureHash = m_activeDrawCallState.materialData.colorTextures[0].getImageHash();
          // Allow vertex colors for textures explicitly in the allowBakedLightingTextures list
          const bool isExplicitlyAllowed = lookupHash(RtxOptions::allowBakedLightingTextures(), textureHash);
          // Also allow vertex colors for decal textures since they need proper vertex color/weighting for blending
          const bool isDecalTexture = lookupHash(RtxOptions::decalTextures(), textureHash) ||
                                      lookupHash(RtxOptions::dynamicDecalTextures(), textureHash) ||
                                      lookupHash(RtxOptions::singleOffsetDecalTextures(), textureHash) ||
                                      lookupHash(RtxOptions::nonOffsetDecalTextures(), textureHash);
          // Also allow vertex colors for particle textures since they need proper vertex color/weighting for effects
          const bool isParticleTexture = lookupHash(RtxOptions::particleTextures(), textureHash);
          if (isExplicitlyAllowed || isDecalTexture || isParticleTexture) {
            targetBuffer = &geoData.color0Buffer;
          }
        }
        break;
      }

      if (targetBuffer != nullptr) {
        assert(!targetBuffer->defined());

        // Only do once for each stream
        if (!streamCopies[element.Stream].defined()) {
          // Deep clonning a buffer object is not cheap (320 bytes to copy and other work). Set a min-size threshold.
          const uint32_t kMinSizeToClone = 512;

          // Check if buffer is actualy a d3d9 orphan
          const bool isOrphan = !(ctx.buffer.getSliceHandle() == ctx.mappedSlice);
          const bool canUseBuffer = ctx.canUseBuffer && m_forceGeometryCopy == false;

          if (canUseBuffer && !isOrphan) {
            // Use the buffer directly if it is not an orphan
            if (ctx.pVBO != nullptr && ctx.pVBO->NeedsUpload())
              m_parent->FlushBuffer(ctx.pVBO);

            streamCopies[element.Stream] = ctx.buffer.subSlice(vertexOffset, numVertexBytes);
          } else if (canUseBuffer && numVertexBytes > kMinSizeToClone) {
            // Create a clone for the orphaned physical slice
            auto clone = ctx.buffer.buffer()->clone();
            clone->rename(ctx.mappedSlice);
            streamCopies[element.Stream] = DxvkBufferSlice(clone, ctx.buffer.offset() + vertexOffset, numVertexBytes);
          } else {
            streamCopies[element.Stream] = m_rtStagingData.alloc(CACHE_LINE_SIZE, numVertexBytes);

            // Acquire prevents the staging allocator from re-using this memory
            streamCopies[element.Stream].buffer()->acquire(DxvkAccess::Read);

            memcpy(streamCopies[element.Stream].mapPtr(0), (uint8_t*) ctx.mappedSlice.mapPtr + vertexOffset, numVertexBytes);
          }
        }

        *targetBuffer = RasterBuffer(streamCopies[element.Stream], element.Offset, ctx.stride, DecodeDecltype(D3DDECLTYPE(element.Type)));
        assert(targetBuffer->offset() % 4 == 0);
      }
    }
  }

  bool D3D9Rtx::processRenderState() {
    DrawCallTransforms& transformData = m_activeDrawCallState.transformData;

    // When games use vertex shaders, the object to world transforms can be unreliable, and so we can ignore them.
    const bool useObjectToWorldTransform = !m_parent->UseProgrammableVS() || (m_parent->UseProgrammableVS() && useVertexCapture() && useWorldMatricesForShaders());
    transformData.objectToWorld = useObjectToWorldTransform ? d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)] : Matrix4();

    transformData.worldToView = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
    transformData.viewToProjection = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

    // NV-DXVK start: Extract camera matrices from vertex shader constants when using programmable VS
    // This is needed for games that use vertex shaders for transforms instead of fixed-function
    //
    // ONE CLEAR PATH: Shader analysis tells us registers, cache from SetVertexShaderConstantF gives us values
    // - No fallbacks, no guessing - deterministic extraction
    // - If this shader has valid matrix registers, extract and cache
    // - If not, use cached camera from a previous shader that did have matrices
    bool extractedFromShader = false;
    static Matrix4 s_cachedView;
    static Matrix4 s_cachedProj;
    static bool s_hasCachedCamera = false;
    static bool s_hasReliableProj = false;  // True if projection was extracted via Z-translation (more accurate)
    static int s_cachedViewReg = -1;  // Remember which registers worked
    static int s_cachedProjReg = -1;
    static uint32_t s_cameraExtractLogCount = 0;

    // UNLIMITED per-frame logging
    static uint32_t s_lastDebugFrame = UINT32_MAX;
    uint32_t currentDebugFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    bool logThisFrame = (currentDebugFrame != s_lastDebugFrame);
    if (logThisFrame) s_lastDebugFrame = currentDebugFrame;

    if (m_parent->UseProgrammableVS()) {
      // Read transforms directly from D3D9 state (not our cache, which may be empty if SetTransform wasn't called)
      const Matrix4& ffView = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
      const Matrix4& ffProj = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];
      const Matrix4& ffWorld = d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)];

      bool ffViewValid = ffView[0][0] != 0.0f || ffView[1][1] != 0.0f || ffView[2][2] != 0.0f;
      bool ffProjValid = ffProj[0][0] != 0.0f || ffProj[1][1] != 0.0f || ffProj[2][2] != 0.0f;
      bool ffWorldValid = ffWorld[0][0] != 0.0f || ffWorld[1][1] != 0.0f || ffWorld[2][2] != 0.0f;

      if (logThisFrame) {
        Logger::info(str::format("[EXTRACT-DEBUG] Frame=", currentDebugFrame, " UseProgrammableVS=TRUE"));
        Logger::info(str::format("[EXTRACT-DEBUG] Fixed-function transforms: ffViewValid=", ffViewValid, " ffProjValid=", ffProjValid, " ffWorldValid=", ffWorldValid));
        if (ffViewValid) {
          Logger::info(str::format("[EXTRACT-DEBUG] ffView row0=[", ffView[0][0], ",", ffView[0][1], ",", ffView[0][2], ",", ffView[0][3], "]"));
          Logger::info(str::format("[EXTRACT-DEBUG] ffView row3=[", ffView[3][0], ",", ffView[3][1], ",", ffView[3][2], ",", ffView[3][3], "]"));
        }
        if (ffProjValid) {
          Logger::info(str::format("[EXTRACT-DEBUG] ffProj row0=[", ffProj[0][0], ",", ffProj[0][1], ",", ffProj[0][2], ",", ffProj[0][3], "]"));
        }
      }

      // Validation: check matrix is not all zeros and has finite values
      auto isValidMatrix = [](const Matrix4& m) -> bool {
        bool allZero = true;
        for (int i = 0; i < 4 && allZero; i++) {
          if (m[i][0] != 0.0f || m[i][1] != 0.0f || m[i][2] != 0.0f || m[i][3] != 0.0f)
            allZero = false;
        }
        if (allZero) return false;
        for (int i = 0; i < 4; i++) {
          if (!std::isfinite(m[i][0]) || !std::isfinite(m[i][1]) ||
              !std::isfinite(m[i][2]) || !std::isfinite(m[i][3])) return false;
        }
        return true;
      };

      // ONE CLEAR PATH: Get matrix from cache OR read directly from D3D9 state
      // The cache may be empty if game sets constants in a different order than expected
      // Direct reading ensures we get the actual current values
      auto getMatrixFromCache = [this, logThisFrame](int reg) -> std::pair<Matrix4, bool> {
        if (reg < 0 || reg >= 256) return {Matrix4(), false};

        // First try the cache
        if (g_hasMatrixAtRegister[reg]) {
          return {g_cachedMatricesByRegister[reg], true};
        }

        // Cache miss - try reading directly from D3D9 VS constant state
        // This handles cases where the game sets constants but our cache wasn't populated
        float matrixData[16];
        const auto& vsConsts = d3d9State().vsConsts;
        if (reg + 4 <= 256) {  // Need 4 consecutive registers for a 4x4 matrix
          // Read 4 float4 registers (c[reg] through c[reg+3])
          for (int i = 0; i < 4; i++) {
            const auto& vec = vsConsts.fConsts[reg + i];
            matrixData[i*4 + 0] = vec.x;
            matrixData[i*4 + 1] = vec.y;
            matrixData[i*4 + 2] = vec.z;
            matrixData[i*4 + 3] = vec.w;
          }

          Matrix4 directMatrix(
            matrixData[0], matrixData[1], matrixData[2], matrixData[3],
            matrixData[4], matrixData[5], matrixData[6], matrixData[7],
            matrixData[8], matrixData[9], matrixData[10], matrixData[11],
            matrixData[12], matrixData[13], matrixData[14], matrixData[15]
          );

          // Check if it's a valid (non-zero, non-identity) matrix
          float sum = std::abs(matrixData[0]) + std::abs(matrixData[5]) + std::abs(matrixData[10]);

          // DEBUG: Always log what we found for key registers (NO LIMIT)
          if (logThisFrame && (reg == 0 || reg == 4 || reg == 48)) {
            Logger::info(str::format("[MATRIX-D3D9-STATE] c", reg, " sum=", sum, " valid=", (sum > 0.01f),
              "\n  row0=[", matrixData[0], ",", matrixData[1], ",", matrixData[2], ",", matrixData[3], "]",
              "\n  row1=[", matrixData[4], ",", matrixData[5], ",", matrixData[6], ",", matrixData[7], "]",
              "\n  row3=[", matrixData[12], ",", matrixData[13], ",", matrixData[14], ",", matrixData[15], "]"));
          }

          if (sum > 0.01f) {
            // Cache it for future use
            g_cachedMatricesByRegister[reg] = directMatrix;
            g_hasMatrixAtRegister[reg] = true;
            return {directMatrix, true};
          }
        }

        return {Matrix4(), false};
      };

      // First, try to get matrix registers from current shader's analysis
      int viewReg = -1;
      int projReg = -1;
      int wvpReg = -1;  // viewProj or worldViewProj - need to derive projection from this
      int worldReg = -1;  // World matrix register for objectToWorld transform
      static uint32_t s_shaderLookupLogCount = 0;

      // PRIMARY: Use pre-scanned defaults from decompiled_shaders directory
      // These are available immediately at startup - no async delay
      if (auto* compatManager = m_shaderCompatibilityManager.get()) {
        if (compatManager->hasDefaultMatrixRegisters()) {
          compatManager->getDefaultMatrixRegisters(viewReg, projReg, worldReg, wvpReg);

          static bool s_loggedDefaults = false;
          if (!s_loggedDefaults) {
            Logger::info(str::format("[MATRIX-REGS] Using pre-scanned defaults: view=c", viewReg,
              " proj=c", projReg, " world=c", worldReg, " wvp=c", wvpReg));
            s_loggedDefaults = true;
          }
        }
      }

      // ONE CLEAR PATH: Extract all matrices from cache
      // Shader analysis tells us registers, SetVertexShaderConstantF populates the cache
      if (viewReg >= 0 || wvpReg >= 0) {
        auto [viewMatrix, hasView] = getMatrixFromCache(viewReg);
        auto [viewProjMatrix, hasViewProj] = getMatrixFromCache(wvpReg);
        auto [projMatrix, hasProj] = getMatrixFromCache(projReg);

        if (logThisFrame) {
          Logger::info(str::format("[MATRIX-EXTRACT] viewReg=c", viewReg, " hasView=", hasView,
            " wvpReg=c", wvpReg, " hasViewProj=", hasViewProj,
            " projReg=c", projReg, " hasProj=", hasProj));
        }

        // View matrix extraction
        // Some games store camera position separately at c8 (vs_worldCamPos)
        // If view matrix has no translation but we have cached camera position, use it
        // Check if view matrix is TRUE identity (all 4 rows must match identity)
        // row0=[1,0,0,0], row1=[0,1,0,0], row2=[0,0,1,0], row3=[0,0,0,1]
        // IMPORTANT: row3 must be (0,0,0,1) - any Z translation means it's a real view transform!
        bool isIdentityLikeView = (
          // Row 0: [1, 0, 0, 0]
          std::abs(viewMatrix[0][0] - 1.0f) < 0.01f &&
          std::abs(viewMatrix[0][1]) < 0.01f &&
          std::abs(viewMatrix[0][2]) < 0.01f &&
          std::abs(viewMatrix[0][3]) < 0.01f &&
          // Row 1: [0, 1, 0, 0]
          std::abs(viewMatrix[1][0]) < 0.01f &&
          std::abs(viewMatrix[1][1] - 1.0f) < 0.01f &&
          std::abs(viewMatrix[1][2]) < 0.01f &&
          std::abs(viewMatrix[1][3]) < 0.01f &&
          // Row 2: [0, 0, 1, 0]
          std::abs(viewMatrix[2][0]) < 0.01f &&
          std::abs(viewMatrix[2][1]) < 0.01f &&
          std::abs(viewMatrix[2][2] - 1.0f) < 0.01f &&
          std::abs(viewMatrix[2][3]) < 0.01f &&
          // Row 3: [0, 0, 0, 1] - translation must be zero!
          std::abs(viewMatrix[3][0]) < 0.01f &&
          std::abs(viewMatrix[3][1]) < 0.01f &&
          std::abs(viewMatrix[3][2]) < 0.01f &&
          std::abs(viewMatrix[3][3] - 1.0f) < 0.01f
        );

        if (hasView && isValidMatrix(viewMatrix) && !isIdentityLikeView) {
          // Real view matrix with camera transform
          Matrix4 finalView = viewMatrix;

          // Check if view matrix has minimal translation and we have camera position
          float viewTransMag = std::sqrt(viewMatrix[3][0]*viewMatrix[3][0] +
                                         viewMatrix[3][1]*viewMatrix[3][1] +
                                         viewMatrix[3][2]*viewMatrix[3][2]);

          if (viewTransMag < 10.0f && g_hasCachedCameraPosition) {
            // View matrix is rotation-only, incorporate camera position from c8
            // View = Rotation * Translate(-camPos)
            Vector3 camPos(g_cachedCameraPosition.x, g_cachedCameraPosition.y, g_cachedCameraPosition.z);

            // Build translation matrix for -camPos
            Matrix4 translateNeg(
              1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              -camPos.x, -camPos.y, -camPos.z, 1
            );

            // Final view = rotation (from c4) * translate(-camPos)
            finalView = viewMatrix * translateNeg;

            if (logThisFrame) {
              Logger::info(str::format("[VIEW-CAMPOS] Using camera pos from c8: (", camPos.x, ", ", camPos.y, ", ", camPos.z, ")",
                "\n  Original view trans: ", viewTransMag,
                "\n  Final view row3: ", finalView[3]));
            }
          }

          transformData.worldToView = finalView;
          s_cachedView = finalView;
          s_cachedViewReg = viewReg;
          extractedFromShader = true;
        } else if (isIdentityLikeView && hasViewProj && isValidMatrix(viewProjMatrix)) {
          // View is identity - use identity view matrix
          // This is common when game uses combined viewProj at c0 without separate view matrix
          transformData.worldToView = Matrix4(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
          );
          s_cachedView = transformData.worldToView;
          s_cachedViewReg = viewReg;
          extractedFromShader = true;

          if (logThisFrame) {
            Logger::info("[VIEW-IDENTITY] Using identity view matrix (view register contains identity)");
          }
        }

        // Projection extraction - either direct, from viewProj, or use viewProj directly
        bool gotProjection = false;
        if (hasProj && isValidMatrix(projMatrix)) {
          // Direct projection matrix available
          transformData.viewToProjection = projMatrix;
          s_cachedProj = projMatrix;
          s_cachedProjReg = projReg;
          gotProjection = true;
        } else if (hasViewProj && isValidMatrix(viewProjMatrix)) {
          // No direct projection - derive from viewProj

          // Check if view is identity-like - if so, viewProj IS the projection matrix
          // viewProj = view * proj, and if view = identity, then viewProj = proj
          if (isIdentityLikeView || !extractedFromShader) {
            // Use viewProj directly as projection since view is identity
            transformData.viewToProjection = viewProjMatrix;
            s_cachedProj = viewProjMatrix;
            s_cachedProjReg = wvpReg;
            gotProjection = true;

            if (logThisFrame) {
              Logger::info(str::format("[PROJ-DIRECT] Using viewProj directly as projection (view is identity)",
                "\n  viewProj row0=", viewProjMatrix[0],
                "\n  viewProj row1=", viewProjMatrix[1],
                "\n  viewProj row2=", viewProjMatrix[2],
                "\n  viewProj row3=", viewProjMatrix[3]));
            }
          } else {
            // View is not identity - need to extract projection from viewProj
            // viewProj = view * proj, so proj = inverse(view) * viewProj
            // IMPORTANT: Use the ORIGINAL viewMatrix, not the camera-position-modified one!
            // The viewProj was built as (raw_view * proj), so we need inverse(raw_view)

            // Check if RAW view is a pure Z-translation matrix: [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,tz,1]
            // This is common in games where camera is offset along view axis
            float tz = viewMatrix[3][2];  // Z-translation from ORIGINAL view matrix
            bool isZTranslation = (
              std::abs(viewMatrix[0][0] - 1.0f) < 0.01f &&
              std::abs(viewMatrix[1][1] - 1.0f) < 0.01f &&
              std::abs(viewMatrix[2][2] - 1.0f) < 0.01f &&
              std::abs(viewMatrix[3][3] - 1.0f) < 0.01f &&
              std::abs(viewMatrix[3][0]) < 0.01f &&
              std::abs(viewMatrix[3][1]) < 0.01f
            );

            Matrix4 extractedProj;

            if (isZTranslation && std::abs(tz) > 0.001f) {
              // View is a Z-translation matrix with tz offset
              // inverse(view) has [3][2] = -tz
              // proj = inverse(view) * viewProj
              // For Z-translation: proj[i] = viewProj[i] for i<3, proj[3] = viewProj[3] - tz*viewProj[2]

              // Get actual screen aspect ratio and correct scaleX to match
              // The game's projection may be built for a different internal aspect ratio
              float screenAspect = (m_activePresentParams && m_activePresentParams->BackBufferHeight > 0)
                ? (float)m_activePresentParams->BackBufferWidth / (float)m_activePresentParams->BackBufferHeight
                : 16.0f / 9.0f;

              // Correct scaleX based on actual screen aspect ratio
              // Keep scaleY (vertical FOV), recalculate scaleX = scaleY / screenAspect
              float scaleY = std::abs(viewProjMatrix[1][1]);
              float correctedScaleX = scaleY / screenAspect;

              extractedProj = Matrix4(
                correctedScaleX, viewProjMatrix[0][1], viewProjMatrix[0][2], viewProjMatrix[0][3],
                viewProjMatrix[1][0], viewProjMatrix[1][1], viewProjMatrix[1][2], viewProjMatrix[1][3],
                viewProjMatrix[2][0], viewProjMatrix[2][1], viewProjMatrix[2][2], viewProjMatrix[2][3],
                viewProjMatrix[3][0] - tz*viewProjMatrix[2][0],
                viewProjMatrix[3][1] - tz*viewProjMatrix[2][1],
                viewProjMatrix[3][2] - tz*viewProjMatrix[2][2],
                viewProjMatrix[3][3] - tz*viewProjMatrix[2][3]
              );

              // Mark this as reliable since Z-translation extraction is accurate
              s_hasReliableProj = true;

              if (logThisFrame) {
                Logger::info(str::format("[PROJ-EXTRACTED] Extracted from Z-translation view (tz=", tz, ") - RELIABLE",
                  "\n  screenAspect=", screenAspect, " origScaleX=", viewProjMatrix[0][0], " correctedScaleX=", correctedScaleX,
                  "\n  proj row0=", extractedProj[0],
                  "\n  proj row1=", extractedProj[1],
                  "\n  proj row2=", extractedProj[2],
                  "\n  proj row3=", extractedProj[3]));
              }
            } else if (s_hasReliableProj) {
              // We have a reliable projection from a previous Z-translation extraction
              // Use it instead of constructing a potentially wrong one
              extractedProj = s_cachedProj;

              if (logThisFrame) {
                Logger::info(str::format("[PROJ-REUSED] Using cached reliable projection from Z-translation",
                  "\n  proj row0=", extractedProj[0],
                  "\n  proj row1=", extractedProj[1]));
              }
            } else {
              // General case: view is a rotation matrix, need to extract proj properly
              // For viewProj = view * proj where view is orthonormal rotation:
              // viewProj column length = proj scale (since rotation preserves length)
              // scaleX = length(viewProj column 0), scaleY = length(viewProj column 1)

              // Column 0: [viewProj[0][0], viewProj[1][0], viewProj[2][0]]
              float col0LenSq = viewProjMatrix[0][0]*viewProjMatrix[0][0] +
                                viewProjMatrix[1][0]*viewProjMatrix[1][0] +
                                viewProjMatrix[2][0]*viewProjMatrix[2][0];
              // Column 1: [viewProj[0][1], viewProj[1][1], viewProj[2][1]]
              float col1LenSq = viewProjMatrix[0][1]*viewProjMatrix[0][1] +
                                viewProjMatrix[1][1]*viewProjMatrix[1][1] +
                                viewProjMatrix[2][1]*viewProjMatrix[2][1];

              float scaleX = std::sqrt(col0LenSq);
              float scaleY = std::sqrt(col1LenSq);

              // Sanity bounds
              if (scaleX < 0.1f || scaleX > 100.0f) scaleX = 1.7f;
              if (scaleY < 0.1f || scaleY > 100.0f) scaleY = 2.4f;

              // Extract near/far from viewProj row 2 and row 3
              // viewProj[2][2] = view_rotated * [0,0,zf/(zf-zn),0] -> length = zf/(zf-zn)
              // viewProj[3][2] = translation contribution - harder to extract
              // Use reasonable defaults but try to get near from viewProj[3][2]/viewProj[2][2]
              float nearPlane = 0.1f;
              float farPlane = 10000.0f;

              // For row2 column2, we need the length of column 2 which has the z scale
              float col2LenSq = viewProjMatrix[0][2]*viewProjMatrix[0][2] +
                                viewProjMatrix[1][2]*viewProjMatrix[1][2] +
                                viewProjMatrix[2][2]*viewProjMatrix[2][2];
              float zScale = std::sqrt(col2LenSq);  // = far/(far-near)

              // row3 col2 contains: -(near*far)/(far-near) transformed by view translation
              // For simplicity, estimate from the ratio if zScale is reasonable
              if (zScale > 0.5f && zScale < 10.0f) {
                // zScale = far/(far-near), typical for far=10000, near=0.1 -> zScale ~= 1.00001
                // Can't easily extract near/far without more info, keep defaults
              }

              extractedProj = Matrix4(
                scaleX, 0, 0, 0,
                0, scaleY, 0, 0,
                0, 0, farPlane / (farPlane - nearPlane), 1,
                0, 0, -nearPlane * farPlane / (farPlane - nearPlane), 0
              );

              if (logThisFrame) {
                Logger::info(str::format("[PROJ-CONSTRUCTED] scaleX=", scaleX, " scaleY=", scaleY,
                  " from col0Len=", std::sqrt(col0LenSq), " col1Len=", std::sqrt(col1LenSq),
                  " viewProj[0][0]=", viewProjMatrix[0][0], " viewProj[1][1]=", viewProjMatrix[1][1]));
              }
            }

            transformData.viewToProjection = extractedProj;
            s_cachedProj = extractedProj;
            s_cachedProjReg = wvpReg;
            gotProjection = true;
          }
        }

        // If extraction succeeded, update cache flag
        // If extraction failed BUT we have previously cached camera, use that
        if (extractedFromShader || gotProjection) {
          s_hasCachedCamera = true;
        } else if (s_hasCachedCamera) {
          // Cache was empty this frame but we have previously cached camera - use it
          transformData.worldToView = s_cachedView;
          transformData.viewToProjection = s_cachedProj;
          extractedFromShader = true;
          gotProjection = true;
        }

        if (logThisFrame) {
          Logger::info(str::format("[CAMERA-EXTRACTED] extractedView=", extractedFromShader, " gotProj=", gotProjection,
            " usedCached=", s_hasCachedCamera,
            "\n  view row3=", transformData.worldToView[3],
            "\n  proj row0=", transformData.viewToProjection[0]));
        }
      }

      // Extract world matrix for per-object transform (changes per draw call)
      if (worldReg >= 0) {
        auto [worldMatrix, hasWorld] = getMatrixFromCache(worldReg);

        if (hasWorld && isValidMatrix(worldMatrix)) {
          // Check if this is an aspect-ratio-baked scale matrix
          // Some games (like Lego Batman 2) bake aspect ratio into the world matrix
          // e.g., X=1.4, Y=0.79 gives ratio 1.77 ≈ 16:9
          // RTX Remix's projection already handles aspect ratio, so we need to remove it

          float scaleX = std::abs(worldMatrix[0][0]);
          float scaleY = std::abs(worldMatrix[1][1]);
          float scaleZ = std::abs(worldMatrix[2][2]);

          // Check if this is a scale-only matrix (no rotation in off-diagonals)
          bool isScaleOnly =
            std::abs(worldMatrix[0][1]) < 0.001f && std::abs(worldMatrix[0][2]) < 0.001f &&
            std::abs(worldMatrix[1][0]) < 0.001f && std::abs(worldMatrix[1][2]) < 0.001f &&
            std::abs(worldMatrix[2][0]) < 0.001f && std::abs(worldMatrix[2][1]) < 0.001f;

          // Check if translation is zero
          bool noTranslation =
            std::abs(worldMatrix[3][0]) < 0.001f &&
            std::abs(worldMatrix[3][1]) < 0.001f &&
            std::abs(worldMatrix[3][2]) < 0.001f;

          // Check if X/Y ratio matches common aspect ratios (16:9 ≈ 1.77, 4:3 ≈ 1.33)
          float xyRatio = (scaleY > 0.001f) ? scaleX / scaleY : 0.0f;
          bool isAspectRatioScale = (xyRatio > 1.7f && xyRatio < 1.85f) ||  // 16:9
                                    (xyRatio > 1.28f && xyRatio < 1.38f);   // 4:3

          Matrix4 finalWorld = worldMatrix;

          // DISABLED: Testing if world aspect fix causes the oval
          // The game might need this aspect scaling for geometry positioning
          #if 0
          if (isScaleOnly && noTranslation && isAspectRatioScale) {
            finalWorld = Matrix4(
              1.0f, 0.0f, 0.0f, 0.0f,
              0.0f, 1.0f, 0.0f, 0.0f,
              0.0f, 0.0f, 1.0f, 0.0f,
              0.0f, 0.0f, 0.0f, 1.0f
            );

            static uint32_t s_aspectFixLogCount = 0;
            if (s_aspectFixLogCount++ < 50) {
              Logger::info(str::format("[WORLD-ASPECT-FIX] Removed aspect scale: X=", scaleX, " Y=", scaleY, " ratio=", xyRatio));
            }
          }
          #endif

          transformData.objectToWorld = finalWorld;

          static uint32_t s_worldLogCount = 0;
          if (s_worldLogCount++ < 20) {
            Logger::info(str::format("[WORLD-MATRIX] Extracted from c", worldReg,
              "\n  row0=", worldMatrix[0],
              "\n  row3=", worldMatrix[3]));
          }
        }
      }

      // FALLBACK: If shader constant extraction failed, use fixed-function transforms
      // The game may use identity matrices in D3D9 state but have real camera elsewhere
      if (!extractedFromShader && ffViewValid && !isIdentityExact(ffView)) {
        transformData.worldToView = ffView;
        extractedFromShader = true;
        if (logThisFrame) {
          Logger::info(str::format("[FALLBACK] Using fixed-function VIEW matrix"));
        }
      }

      if (ffProjValid && !isIdentityExact(ffProj)) {
        // Check if projection looks real (not identity)
        if (ffProj[0][0] != 1.0f || ffProj[1][1] != 1.0f) {
          transformData.viewToProjection = ffProj;
          if (logThisFrame) {
            Logger::info(str::format("[FALLBACK] Using fixed-function PROJECTION matrix"));
          }
        }
      }

    }

    // NV-DXVK end

    transformData.objectToView = transformData.worldToView * transformData.objectToWorld;

    // NV-DXVK start: UNLIMITED per-frame logging
    static uint32_t s_lastLoggedFrame = UINT32_MAX;
    uint32_t currentFrame = m_parent->GetDXVKDevice()->getCurrentFrameId();
    if (currentFrame != s_lastLoggedFrame) {
      s_lastLoggedFrame = currentFrame;
      Logger::info(str::format("[FRAME-MATRICES] Frame=", currentFrame,
        " extracted=", extractedFromShader,
        "\n  VIEW row0=[", transformData.worldToView[0][0], ",", transformData.worldToView[0][1], ",", transformData.worldToView[0][2], ",", transformData.worldToView[0][3], "]",
        "\n  VIEW row3=[", transformData.worldToView[3][0], ",", transformData.worldToView[3][1], ",", transformData.worldToView[3][2], ",", transformData.worldToView[3][3], "]",
        "\n  PROJ row0=[", transformData.viewToProjection[0][0], ",", transformData.viewToProjection[0][1], ",", transformData.viewToProjection[0][2], ",", transformData.viewToProjection[0][3], "]",
        "\n  PROJ row1=[", transformData.viewToProjection[1][0], ",", transformData.viewToProjection[1][1], ",", transformData.viewToProjection[1][2], ",", transformData.viewToProjection[1][3], "]",
        "\n  WORLD row0=[", transformData.objectToWorld[0][0], ",", transformData.objectToWorld[0][1], ",", transformData.objectToWorld[0][2], ",", transformData.objectToWorld[0][3], "]"));
    }
    // NV-DXVK end

    // Some games pass invalid matrices which D3D9 apparently doesnt care about.
    // since we'll be doing inversions and other matrix operations, we need to 
    // sanitize those or there be nans.
    transformData.sanitize();

    if (m_flags.test(D3D9RtxFlag::DirtyClipPlanes)) {
      m_flags.clr(D3D9RtxFlag::DirtyClipPlanes);

      // Find one truly enabled clip plane because we don't support more than one
      transformData.enableClipPlane = false;
      if (d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] != 0) {
        for (int i = 0; i < caps::MaxClipPlanes; ++i) {
          // Check the enable bit
          if ((d3d9State().renderStates[D3DRS_CLIPPLANEENABLE] & (1 << i)) == 0)
            continue;

          // Make sure that the plane equation is not degenerate
          const Vector4 plane = Vector4(d3d9State().clipPlanes[i].coeff);
          if (lengthSqr(plane.xyz()) > 0.f) {
            if (transformData.enableClipPlane) {
              ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Using more than 1 user clip plane is not supported.")));
              break;
            }

            transformData.enableClipPlane = true;
            transformData.clipPlane = plane;
          }
        }
      }
    }

    if (m_flags.test(D3D9RtxFlag::DirtyLights)) {
      m_flags.clr(D3D9RtxFlag::DirtyLights);

      std::vector<D3DLIGHT9> activeLightsRT;
      uint32_t lightIdx = 0;
      for (auto idx : d3d9State().enabledLightIndices) {
        if (idx == UINT32_MAX)
          continue;
        activeLightsRT.push_back(d3d9State().lights[idx].value());
      }

      m_parent->EmitCs([activeLightsRT, lightIdx](DxvkContext* ctx) {
          static_cast<RtxContext*>(ctx)->addLights(activeLightsRT.data(), activeLightsRT.size());
        });
    }

    // Stencil state is important to Remix
    m_activeDrawCallState.stencilEnabled = d3d9State().renderStates[D3DRS_STENCILENABLE];

    // Process textures
    if (m_parent->UseProgrammablePS()) {
      return processTextures<false>();
    } else {
      return processTextures<true>();
    }
  }

  D3D9Rtx::DrawCallType D3D9Rtx::makeDrawCallType(const DrawContext& drawContext) {
    // Track the drawcall index so we can use it in rtx_context
    m_activeDrawCallState.drawCallID = m_drawCallID++;
    m_activeDrawCallState.isDrawingToRaytracedRenderTarget = false;
    m_activeDrawCallState.isUsingRaytracedRenderTarget = false;

    if (m_drawCallID < (uint32_t)RtxOptions::drawCallRange().x ||
        m_drawCallID > (uint32_t)RtxOptions::drawCallRange().y) {
      return { RtxGeometryStatus::Ignored, false };
    }

    // Raytraced Render Target Support
    // If the bound texture for this draw call is one that has been used as a render target then store its id
    if (RtxOptions::RaytracedRenderTarget::enable()) {
      for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
        if (lookupHash(RtxOptions::raytracedRenderTargetTextures(), texture->GetImage()->getDescriptorHash())) {
          m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
        }
      }
    }

    if (m_parent->UseProgrammableVS() && !useVertexCapture()) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipping draw call with shader usage as vertex capture is not enabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (drawContext.PrimitiveCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, primitive count was 0."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Only certain draw calls are worth raytracing
    if (!isPrimitiveSupported(drawContext.PrimitiveType)) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace an unsupported primitive topology [", drawContext.PrimitiveType, "]. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!RtxOptions::enableAlphaTest() && m_parent->IsAlphaTestEnabled()) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-tested draw call when alpha-tested objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }

    if (!RtxOptions::enableAlphaBlend() && d3d9State().renderStates[D3DRS_ALPHABLENDENABLE]) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing an alpha-blended draw call when alpha-blended objects disabled in RT. Ignoring.")));
      return { RtxGeometryStatus::Ignored, false };
    }
    
    if (m_activeOcclusionQueries > 0) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to raytrace an occlusion query. Ignoring.")));
      return { RtxGeometryStatus::Rasterized, false };
    }

    if (d3d9State().renderTargets[kRenderTargetIndex] == nullptr) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, as no color render target bound."));
      return { RtxGeometryStatus::Ignored, false };
    }

    constexpr DWORD rgbWriteMask = D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE;
    if ((d3d9State().renderStates[ColorWriteIndex(kRenderTargetIndex)] & rgbWriteMask) != rgbWriteMask) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, colour write disabled."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Ensure present parameters for the swapchain have been cached
    // Note: This assumes that ResetSwapChain has been called at some point before this call, typically done after creating a swapchain.
    assert(m_activePresentParams.has_value());

    // Attempt to detect shadow mask draws and ignore them
    // Conditions: non-textured flood-fill draws into a small quad render target
    if (((d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG1 && d3d9State().textureStages[0][D3DTSS_COLORARG1] != D3DTA_TEXTURE) ||
         (d3d9State().textureStages[0][D3DTSS_COLOROP] == D3DTOP_SELECTARG2 && d3d9State().textureStages[0][D3DTSS_COLORARG2] != D3DTA_TEXTURE))) {
      const auto& rtExt = d3d9State().renderTargets[kRenderTargetIndex]->GetSurfaceExtent();
      // If rt is a quad at least 4 times smaller than backbuffer and the format is invalid format, then it is likely a shadow mask
      if (rtExt.width == rtExt.height && rtExt.width < m_activePresentParams->BackBufferWidth / 4 &&
          Resources::getFormatCompatibilityCategory(d3d9State().renderTargets[kRenderTargetIndex]->GetImageView(false)->imageInfo().format) == RtxTextureFormatCompatibilityCategory::InvalidFormatCompatibilityCategory) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped shadow mask drawcall."));
        return { RtxGeometryStatus::Ignored, false };
      }
    }

    // Raytraced Render Target
    // If this isn't the primary render target but we have used this render target before then 
    // store the current camera matrices in case this render target is intended to be used as 
    // a texture for some geometry later
    if (RtxOptions::RaytracedRenderTarget::enable()) {
      D3D9CommonTexture* texture = GetCommonTexture(d3d9State().renderTargets[kRenderTargetIndex]->GetBaseTexture());
      if (texture && lookupHash(RtxOptions::raytracedRenderTargetTextures(), texture->GetImage()->getDescriptorHash())) {
        m_activeDrawCallState.isDrawingToRaytracedRenderTarget = true;
        return { RtxGeometryStatus::RayTraced, false };
      }
    }

    if (!s_isDxvkResolutionEnvVarSet) {
      // NOTE: This can fail when setting DXVK_RESOLUTION_WIDTH or HEIGHT
      const auto* rtDesc = d3d9State().renderTargets[kRenderTargetIndex]->GetCommonTexture()->Desc();
      const bool isPrimary = isRenderTargetPrimary(*m_activePresentParams, rtDesc);

      Logger::info(str::format("[RT-PRIMARY-CHECK] BackBuffer=", m_activePresentParams->BackBufferWidth, "x", m_activePresentParams->BackBufferHeight,
                               " RenderTarget=", rtDesc->Width, "x", rtDesc->Height,
                               " PrimCount=", drawContext.PrimitiveCount,
                               " PrimType=", (int)drawContext.PrimitiveType,
                               " isPrimary=", isPrimary ? "YES" : "NO"));

      if (!isPrimary) {
        Logger::info(str::format("[RT-RASTERIZE-FALLBACK] Rasterizing ", drawContext.PrimitiveCount, " primitives to ", rtDesc->Width, "x", rtDesc->Height));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    // Detect stencil shadow draws and ignore them
    // Conditions: passingthrough stencil is enabled with increment or decrement z-fail action
    if (d3d9State().renderStates[D3DRS_STENCILENABLE] == TRUE &&
        d3d9State().renderStates[D3DRS_STENCILFUNC] == D3DCMP_ALWAYS &&
        (d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECR || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCR ||
         d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_DECRSAT || d3d9State().renderStates[D3DRS_STENCILZFAIL] == D3DSTENCILOP_INCRSAT) &&
        d3d9State().renderStates[D3DRS_ZWRITEENABLE] == FALSE) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped stencil shadow drawcall."));
      return { RtxGeometryStatus::Ignored, false };
    }

    // Check UI only to the primary render target
    if (isRenderingUI()) {
      return {
        RtxGeometryStatus::Rasterized,
        true, // UI rendering detected => trigger RTX injection
      };
    }

    // TODO(REMIX-760): Support reverse engineering pre-transformed vertices
    if (d3d9State().vertexDecl != nullptr) {
      if (d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasPositionT)) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped drawcall, using pre-transformed vertices which isn't currently supported."));
        return { RtxGeometryStatus::Rasterized, false };
      }
    }

    return { RtxGeometryStatus::RayTraced, false };
  }

  bool D3D9Rtx::checkBoundTextureCategory(const fast_unordered_set& textureCategory) const {
    const uint32_t usedSamplerMask = m_parent->m_psShaderMasks.samplerMask | m_parent->m_vsShaderMasks.samplerMask;
    const uint32_t usedTextureMask = m_parent->m_activeTextures & usedSamplerMask;
    for (uint32_t idx : bit::BitMask(usedTextureMask)) {
      if (!d3d9State().textures[idx])
        continue;

      auto texture = GetCommonTexture(d3d9State().textures[idx]);

      const XXH64_hash_t texHash = texture->GetSampleView(false)->image()->getHash();
      if (textureCategory.find(texHash) != textureCategory.end()) {
        return true;
      }
    }

    return false;
  }

  bool D3D9Rtx::isRenderingUI() {
    if (!m_parent->UseProgrammableVS() && orthographicIsUI()) {
      // Here we assume drawcalls with an orthographic projection are UI calls (as this pattern is common, and we can't raytrace these objects).
      const bool isOrthographic = (d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)][3][3] == 1.0f);
      const bool zWriteEnabled = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
      if (isOrthographic && !zWriteEnabled) {
        return true;
      }
    }

    // Check if UI texture bound
    return checkBoundTextureCategory(RtxOptions::uiTextures());
  }

  PrepareDrawFlags D3D9Rtx::internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext) {
    ScopedCpuProfileZone();

    m_activeDrawCallState.forceShaderCapture = false;
    m_activeDrawCallState.originalVertexData.clear();
    m_activeDrawCallState.originalIndexData.clear();
    m_activeDrawCallState.capturedVertexStreams.clear();
    m_activeDrawCallState.capturedVertexElements.clear();

    // Set draw context parameters needed for index buffer capture
    m_activeDrawCallState.originalFirstIndex = drawContext.StartIndex;
    m_activeDrawCallState.originalBaseVertex = drawContext.BaseVertexIndex;

    // RTX was injected => treat everything else as rasterized 
    if (m_rtxInjectTriggered) {
      return RtxOptions::skipDrawCallsPostRTXInjection()
             ? PrepareDrawFlag::Ignore
             : PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    auto drawType = makeDrawCallType(drawContext);
    auto status = drawType.status;
    bool triggerRtxInjection = drawType.triggerRtxInjection;
    const bool forceCaptureAll = ShaderOutputCapturer::captureAllDraws();
    if (status == RtxGeometryStatus::Rasterized && forceCaptureAll) {
      status = RtxGeometryStatus::RayTraced;
      m_activeDrawCallState.forceShaderCapture = true;
    }

    // When raytracing is enabled we want to completely remove the ignored drawcalls from further processing as early as possible
    const PrepareDrawFlags prepareFlagsForIgnoredDraws = RtxOptions::enableRaytracing()
                                                         ? PrepareDrawFlag::Ignore
                                                         : PrepareDrawFlag::PreserveDrawCallAndItsState;

    if (status == RtxGeometryStatus::Ignored) {
      return prepareFlagsForIgnoredDraws;
    }

    if (triggerRtxInjection) {
      // Bind all resources required for this drawcall to context first (i.e. render targets)
      m_parent->PrepareDraw(drawContext.PrimitiveType);

      triggerInjectRTX();

      m_rtxInjectTriggered = true;
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    if (status == RtxGeometryStatus::Rasterized) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_forceGeometryCopy = RtxOptions::useBuffersDirectly() == false;
    m_forceGeometryCopy |= m_parent->GetOptions()->allowDiscard == false;

    // The packet we'll send to RtxContext with information about geometry
    RasterGeometry& geoData = m_activeDrawCallState.geometryData;
    geoData = {};
    geoData.cullMode = DecodeCullMode(D3DCULL(d3d9State().renderStates[D3DRS_CULLMODE]));
    geoData.frontFace = VK_FRONT_FACE_CLOCKWISE;
    geoData.topology = DecodeInputAssemblyState(drawContext.PrimitiveType).primitiveTopology;

    // This can be negative!!
    int vertexIndexOffset = drawContext.BaseVertexIndex;

    // Process index buffer
    uint32_t minIndex = 0, maxIndex = 0;
    if (indexContext.indexType != VK_INDEX_TYPE_NONE_KHR) {
      geoData.indexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);

      // Store the original index count for buffer capture (BEFORE rebasing modifies it)
      m_activeDrawCallState.originalIndexCount = geoData.indexCount;

      if (indexContext.indexType == VK_INDEX_TYPE_UINT16)
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint16_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 2, indexContext.indexType);
      else
        geoData.indexBuffer = RasterBuffer(processIndexBuffer<uint32_t>(geoData.indexCount, drawContext.StartIndex, indexContext, minIndex, maxIndex), 0, 4, indexContext.indexType);

      // Unlikely, but invalid
      if (maxIndex == minIndex) {
        ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no triangles detected in index buffer."));
        return prepareFlagsForIgnoredDraws;
      }

      geoData.vertexCount = maxIndex - minIndex + 1;
      vertexIndexOffset += minIndex;
    } else {
      geoData.vertexCount = GetVertexCount(drawContext.PrimitiveType, drawContext.PrimitiveCount);
    }

    if (geoData.vertexCount == 0) {
      ONCE(Logger::info("[RTX-Compatibility-Info] Skipped invalid drawcall, no vertices detected."));
      return prepareFlagsForIgnoredDraws;
    }

    if (RtxOptions::RaytracedRenderTarget::enable()) {
      // If this draw call has an RT texture bound
      if (m_activeDrawCallState.isUsingRaytracedRenderTarget) {
        // We validate this state below
        m_activeDrawCallState.isUsingRaytracedRenderTarget = false;
        // Try and find the has of the positions
        for (uint32_t i : bit::BitMask(m_parent->GetActiveRTTextures())) {
          D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[i]);
          auto hash = texture->GetImage()->getDescriptorHash();
          if (lookupHash(RtxOptions::raytracedRenderTargetTextures(), hash)) {
            // Mark this as a valid Raytraced Render Target draw call
            m_activeDrawCallState.isUsingRaytracedRenderTarget = true;
          }
        }
      }
    }

    m_activeDrawCallState.categories = 0;
    m_activeDrawCallState.materialData = {};
    m_activeDrawCallState.capturedD3D9Textures.clear(); // Clear captured textures for new draw call
    m_activeDrawCallState.renderTargetReplacementSlot = -1; // Reset render target replacement slot
    m_activeDrawCallState.originalRenderTargetHash = 0; // Reset original RT hash
    m_activeDrawCallState.depthTextureMask = m_parent->m_depthTextures; // Capture shadow sampler mask

    // Fetch all the legacy state (colour modes, alpha test, etc...)
    setLegacyMaterialState(m_parent, m_parent->m_alphaSwizzleRTs & (1 << kRenderTargetIndex), m_activeDrawCallState.materialData);

    // Fetch fog state 
    setFogState(m_parent, m_activeDrawCallState.fogState);

    // Fetch all the render state and send it to rtx context (textures, transforms, etc.)
    if (!processRenderState()) {
      return prepareFlagsForIgnoredDraws;
    }

    // Max offseted index value within a buffer slice that geoData contains
    const uint32_t maxOffsetedIndex = maxIndex - minIndex;

    // Copy all the vertices into a staging buffer.  Assign fields of the geoData structure.
    processVertices(vertexContext, vertexIndexOffset, geoData);
    geoData.futureGeometryHashes = computeHash(geoData, maxOffsetedIndex);
    geoData.futureBoundingBox = computeAxisAlignedBoundingBox(geoData);
    
    // Process skinning data
    m_activeDrawCallState.futureSkinningData = processSkinning(geoData);

    // Hash material data
    m_activeDrawCallState.materialData.updateCachedHash();

    // For shader based drawcalls we also want to capture the vertex shader output
    const bool needVertexCapture = m_parent->UseProgrammableVS() && useVertexCapture();
    if (needVertexCapture) {
      prepareVertexCapture(vertexIndexOffset);
    }

    m_activeDrawCallState.usesVertexShader = m_parent->UseProgrammableVS();
    m_activeDrawCallState.usesPixelShader = m_parent->UseProgrammablePS();

    if (m_activeDrawCallState.usesVertexShader) {
      m_activeDrawCallState.programmableVertexShaderInfo = d3d9State().vertexShader->GetCommonShader()->GetInfo();
    }
    
    if (m_activeDrawCallState.usesPixelShader) {
      m_activeDrawCallState.programmablePixelShaderInfo = d3d9State().pixelShader->GetCommonShader()->GetInfo();
    }
    
    // SHADER OUTPUT CAPTURE: Capture shader constants for re-execution
    // Access constants directly from D3D9 device state (m_state.vsConsts/psConsts)
    // NOTE: We capture ALL 224 constants, not just maxConstIndexF, because games often
    // set constants beyond what the shader metadata declares (e.g., game sets c[30] but shader only declares using c[0-11])
    if (m_activeDrawCallState.usesVertexShader) {
      constexpr uint32_t kMaxD3D9VSConsts = 256; // D3D9 vertex shader constant limit
      m_activeDrawCallState.vertexShaderConstantData.resize(kMaxD3D9VSConsts);
      const Direct3DState9& state = d3d9State();
      // Fast copy all constants
      std::memcpy(m_activeDrawCallState.vertexShaderConstantData.data(),
                  state.vsConsts.fConsts,
                  kMaxD3D9VSConsts * sizeof(Vector4));
    }
    
    if (m_activeDrawCallState.usesPixelShader) {
      constexpr uint32_t kMaxD3D9PSConsts = 224; // D3D9 pixel shader constant limit
      m_activeDrawCallState.pixelShaderConstantData.resize(kMaxD3D9PSConsts);
      const Direct3DState9& state = d3d9State();
      // Fast copy all constants
      std::memcpy(m_activeDrawCallState.pixelShaderConstantData.data(),
                  state.psConsts.fConsts,
                  kMaxD3D9PSConsts * sizeof(Vector4));
    }
    
    m_activeDrawCallState.cameraType = CameraType::Unknown;

    m_activeDrawCallState.minZ = std::clamp(d3d9State().viewport.MinZ, 0.0f, 1.0f);
    m_activeDrawCallState.maxZ = std::clamp(d3d9State().viewport.MaxZ, 0.0f, 1.0f);

    // Capture viewport and scissor state for shader re-execution
    const auto& vp = d3d9State().viewport;
    m_activeDrawCallState.originalViewport.x = static_cast<float>(vp.X);
    m_activeDrawCallState.originalViewport.y = static_cast<float>(vp.Y);
    m_activeDrawCallState.originalViewport.width = static_cast<float>(vp.Width);
    m_activeDrawCallState.originalViewport.height = static_cast<float>(vp.Height);
    m_activeDrawCallState.originalViewport.minDepth = vp.MinZ;
    m_activeDrawCallState.originalViewport.maxDepth = vp.MaxZ;

    m_activeDrawCallState.originalScissor.offset.x = vp.X;
    m_activeDrawCallState.originalScissor.offset.y = vp.Y;
    m_activeDrawCallState.originalScissor.extent.width = vp.Width;
    m_activeDrawCallState.originalScissor.extent.height = vp.Height;

    m_activeDrawCallState.zWriteEnable = d3d9State().renderStates[D3DRS_ZWRITEENABLE];
    m_activeDrawCallState.zEnable = d3d9State().renderStates[D3DRS_ZENABLE] == D3DZB_TRUE;
    
    // Now that the DrawCallState is complete, we can use heuristics for detection
    m_activeDrawCallState.setupCategoriesForHeuristics(m_seenCameraPositionsPrev.size(),
                                                       m_seenCameraPositions);

    if (RtxOptions::fogIgnoreSky() && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      m_activeDrawCallState.fogState.mode = D3DFOG_NONE;
    }

    // Ignore sky draw calls that are being drawn to a Raytraced Render Target
    // Raytraced Render Target scenes just use the same sky as the main scene, no need to duplicate them
    if (m_activeDrawCallState.isDrawingToRaytracedRenderTarget && m_activeDrawCallState.categories.test(InstanceCategories::Sky)) {
      return prepareFlagsForIgnoredDraws;
    }

    assert(status == RtxGeometryStatus::RayTraced);

    const bool preserveOriginalDraw = needVertexCapture;

    return
      PrepareDrawFlag::CommitToRayTracing |
      (m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_DRAW_CALL_STATE) ? PrepareDrawFlag::ApplyDrawState : 0) |
      (preserveOriginalDraw ? PrepareDrawFlag::PreserveDrawCallAndItsState : 0);
  }

  void D3D9Rtx::triggerInjectRTX() {
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Send command to inject RTX
    m_parent->EmitCs([cReflexFrameId = GetReflexFrameId()](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->injectRTX(cReflexFrameId);
    });
  }

  void D3D9Rtx::CommitGeometryToRT(const DrawContext& drawContext) {
    ScopedCpuProfileZone();
    const uint32_t d3d9InstanceCount = m_parent->GetInstanceCount();
    auto drawInfo = m_parent->GenerateDrawInfo(drawContext.PrimitiveType, drawContext.PrimitiveCount, d3d9InstanceCount);

    // CRITICAL DEBUG: Log instance counts
    static uint32_t commitLogCount = 0;
    if (++commitLogCount <= 20) {
      Logger::info(str::format("[INSTANCE-DEBUG] CommitGeometryToRT #", commitLogCount,
                              " d3d9InstanceCount=", d3d9InstanceCount,
                              " drawInfo.instanceCount=", drawInfo.instanceCount,
                              " indexed=", drawContext.Indexed));
    }

    DrawParameters params;
    params.instanceCount = drawInfo.instanceCount;
    params.vertexOffset = drawContext.BaseVertexIndex;
    params.firstIndex = drawContext.StartIndex;
    // DXVK overloads the vertexCount/indexCount in DrawInfo
    if (drawContext.Indexed) {
      params.indexCount = drawInfo.vertexCount;
    } else {
      params.vertexCount = drawInfo.vertexCount;
    }

    submitActiveDrawCallState();

    m_parent->EmitCs([params, this](DxvkContext* ctx) {
      assert(dynamic_cast<RtxContext*>(ctx));
      DrawCallState drawCallState;
      if (m_drawCallStateQueue.pop(drawCallState)) {
        static_cast<RtxContext*>(ctx)->commitGeometryToRT(params, drawCallState);
      }
    });
  }

  void D3D9Rtx::submitActiveDrawCallState() {
    // We must be prepared for `push` failing here, this can happen, since we're pushing to a circular buffer, which
    //  may not have room for new entries.  In such cases, we trust that the consumer thread will make space for us, and
    //  so we may just need to wait a little bit.
    while (!m_drawCallStateQueue.push(std::move(m_activeDrawCallState))) {
      Sleep(0);
    }
  }

  Future<SkinningData> D3D9Rtx::processSkinning(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    static const auto kEmptySkinningFuture = Future<SkinningData>();

    if (m_parent->UseProgrammableVS()) {
      return kEmptySkinningFuture;
    }

    // Some games set vertex blend without enough data to actually do the blending, handle that logic below.

    const bool hasBlendWeight = d3d9State().vertexDecl != nullptr ? d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasBlendWeight) : false;
    const bool hasBlendIndices = d3d9State().vertexDecl != nullptr ? d3d9State().vertexDecl->TestFlag(D3D9VertexDeclFlag::HasBlendIndices) : false;
    const bool indexedVertexBlend = hasBlendIndices && d3d9State().renderStates[D3DRS_INDEXEDVERTEXBLENDENABLE];

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] == D3DVBF_DISABLE) {
      return kEmptySkinningFuture;
    }

    if (d3d9State().renderStates[D3DRS_VERTEXBLEND] != D3DVBF_0WEIGHTS) {
      if (!hasBlendWeight) {
        return kEmptySkinningFuture;
      }
    } else if (!indexedVertexBlend) {
      return kEmptySkinningFuture;
    }

    // We actually have skinning data now, process it!

    uint32_t numBonesPerVertex = 0;
    switch (d3d9State().renderStates[D3DRS_VERTEXBLEND]) {
    case D3DVBF_0WEIGHTS: numBonesPerVertex = 1; break;
    case D3DVBF_1WEIGHTS: numBonesPerVertex = 2; break;
    case D3DVBF_2WEIGHTS: numBonesPerVertex = 3; break;
    case D3DVBF_3WEIGHTS: numBonesPerVertex = 4; break;
    }

    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery blendIndices;
    // Analyze the vertex data and find the min and max bone indices used in this mesh.
    // The min index is used to detect a case when vertex blend is enabled but there is just one bone used in the mesh,
    // so we can drop the skinning pass. That is processed in RtxContext::commitGeometryToRT(...)
    if (indexedVertexBlend && geoData.blendIndicesBuffer.defined()) {
      auto& buffer = geoData.blendIndicesBuffer;

      blendIndices.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
      blendIndices.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
      blendIndices.stride = buffer.stride();
      blendIndices.size = blendIndices.stride * vertexCount;
      blendIndices.ref = buffer.buffer().ptr();

      // Acquire prevents the staging allocator from re-using this memory
      blendIndices.ref->acquire(DxvkAccess::Read);
      // Make sure we hold on to this reference while the hashing is in flight
      blendIndices.ref->incRef();
    } else {
      blendIndices.ref = nullptr;
    }

    // Copy bones up to the max bone we have registered so far.
    const uint32_t maxBone = m_maxBone > 0 ? m_maxBone : 255;
    const uint32_t startBoneTransform = GetTransformIndex(D3DTS_WORLDMATRIX(0));

    if (m_stagedBonesCount + maxBone >= m_stagedBones.size()) {
      throw DxvkError("Bones temp storage is too small.");
    }

    Matrix4* boneMatrices = m_stagedBones.data() + m_stagedBonesCount;
    memcpy(boneMatrices, d3d9State().transforms.data() + startBoneTransform, sizeof(Matrix4)*(maxBone + 1));
    m_stagedBonesCount += maxBone + 1;

    return m_pGeometryWorkers->Schedule([boneMatrices, blendIndices, numBonesPerVertex, vertexCount]()->SkinningData {
      ScopedCpuProfileZone();
      uint32_t numBones = numBonesPerVertex;

      int minBoneIndex = 0;
      if (blendIndices.ref) {
        const uint8_t* pBlendIndices = blendIndices.pBase;
        // Find out how many bone indices are specified for each vertex.
        // This is needed to find out the min bone index and ignore the padding zeroes.
        int maxBoneIndex = -1;
        if (!getMinMaxBoneIndices(pBlendIndices, blendIndices.stride, vertexCount, numBonesPerVertex, minBoneIndex, maxBoneIndex)) {
          minBoneIndex = 0;
          maxBoneIndex = 0;
        }
        numBones = maxBoneIndex + 1;

        // Release this memory back to the staging allocator
        blendIndices.ref->release(DxvkAccess::Read);
        blendIndices.ref->decRef();
      }

      // Pass bone data to RT back-end

      SkinningData skinningData;
      skinningData.pBoneMatrices.reserve(numBones);

      for (uint32_t n = 0; n < numBones; n++) {
        skinningData.pBoneMatrices.push_back(boneMatrices[n]);
      }

      skinningData.minBoneIndex = minBoneIndex;
      skinningData.numBones = numBones;
      skinningData.numBonesPerVertex = numBonesPerVertex;
      skinningData.computeHash(); // Computes the hash and stores it in the skinningData itself

      return skinningData;
    });
  }

  template<bool FixedFunction>
  bool D3D9Rtx::processTextures() {
    // We don't support full legacy materials in fixed function mode yet..
    // This implementation finds the most relevant textures bound from the
    // following criteria:
    //   - Texture actually bound (and used) by stage
    //   - First N textures bound to a specific texcoord index
    //   - Prefer lowest texcoord index
    // In non-fixed function (shaders), take the first N textures.

    // Used args for a given operation.
    auto ArgsMask = [](DWORD Op) {
      switch (Op) {
      case D3DTOP_DISABLE:
        return 0b000u; // No Args
      case D3DTOP_SELECTARG1:
      case D3DTOP_PREMODULATE:
        return 0b010u; // Arg 1
      case D3DTOP_SELECTARG2:
        return 0b100u; // Arg 2
      case D3DTOP_MULTIPLYADD:
      case D3DTOP_LERP:
        return 0b111u; // Arg 0, 1, 2
      default:
        return 0b110u; // Arg 1, 2
      }
    };

    // Currently we only support 2 textures
    constexpr uint32_t NumTexcoordBins = FixedFunction ? (D3DDP_MAXTEXCOORD * LegacyMaterialData::kMaxSupportedTextures) : LegacyMaterialData::kMaxSupportedTextures;

    bool useStageTextureFactorBlending = true;
    bool useMultipleStageTextureFactorBlending = false;

    // Build a mapping of texcoord indices to stage
    const uint8_t kInvalidStage = 0xFF;
    uint8_t texcoordIndexToStage[NumTexcoordBins];
    if constexpr (FixedFunction) {
      memset(&texcoordIndexToStage[0], kInvalidStage, sizeof(texcoordIndexToStage));
      for (uint32_t stage = 0; stage < caps::TextureStageCount; stage++) {
        auto isTextureFactorBlendingEnabled = [&](const auto& tss) -> bool {
          const auto colorOp = tss[DXVK_TSS_COLOROP];
          const auto alphaOp = tss[DXVK_TSS_ALPHAOP];

          if (colorOp == D3DTOP_DISABLE && alphaOp == D3DTOP_DISABLE)
            return false;

          const auto a1c = tss[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK;
          const auto a2c = tss[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK;
          const auto a1a = tss[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK;
          const auto a2a = tss[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK;

          // If previous stage wrote to TEMP the prior result source this stage
          // should read is D3DTA_TEMP otherwise its D3DTA_CURRENT.
          DWORD prevResultSel = D3DTA_CURRENT;
          if (stage != 0) {
            const auto& prev = d3d9State().textureStages[stage - 1];
            const auto resultArg = prev[DXVK_TSS_RESULTARG] & D3DTA_SELECTMASK;
            prevResultSel = (resultArg == D3DTA_TEMP) ? D3DTA_TEMP : D3DTA_CURRENT;
          }

          auto isModulate = [](DWORD op) {
            return op == D3DTOP_MODULATE || op == D3DTOP_MODULATE2X || op == D3DTOP_MODULATE4X;
          };

          const bool colorMul =
            isModulate(colorOp) &&
            ((a1c == D3DTA_TFACTOR && a2c == prevResultSel) ||
             (a2c == D3DTA_TFACTOR && a1c == prevResultSel));

          const bool alphaMul =
            isModulate(alphaOp) &&
            ((a1a == D3DTA_TFACTOR && a2a == prevResultSel) ||
             (a2a == D3DTA_TFACTOR && a1a == prevResultSel));

          return colorMul || alphaMul;
        };

        // Support texture factor blending besides the first stage. Currently, we only support 1 additional stage tFactor blending.
        // Note: If the tFactor is disabled for current texture (useStageTextureFactorBlending) then we should ignore the multiple stage tFactor blendings.
        bool isCurrentStageTextureFactorBlendingEnabled = false;
        if (useStageTextureFactorBlending &&
            RtxOptions::enableMultiStageTextureFactorBlending() &&
            stage != 0 &&
            isTextureFactorBlendingEnabled(d3d9State().textureStages[stage])) {
          isCurrentStageTextureFactorBlendingEnabled = true;
          useMultipleStageTextureFactorBlending = true;
        }

        if (d3d9State().textures[stage] == nullptr)
          continue;

        const auto& data = d3d9State().textureStages[stage];

        // Subsequent stages do not occur if this is true.
        if (data[DXVK_TSS_COLOROP] == D3DTOP_DISABLE)
          break;

        const std::uint32_t argsMask = ArgsMask(data[DXVK_TSS_COLOROP]) | ArgsMask(data[DXVK_TSS_ALPHAOP]);
        const auto firstTexMask  = ((data[DXVK_TSS_COLORARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG0] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto secondTexMask = ((data[DXVK_TSS_COLORARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG1] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const auto thirdTexMask  = ((data[DXVK_TSS_COLORARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE) || ((data[DXVK_TSS_ALPHAARG2] & D3DTA_SELECTMASK) == D3DTA_TEXTURE);
        const std::uint32_t texMask =
          (firstTexMask  ? 0b001 : 0) |
          (secondTexMask ? 0b010 : 0) |
          (thirdTexMask  ? 0b100 : 0);

        // Is texture used?
        if ((argsMask & texMask) == 0)
          continue;

        D3D9CommonTexture* texture = GetCommonTexture(d3d9State().textures[stage]);

        // Remix can only handle 2D textures - no volumes.
        if (texture->GetType() != D3DRTYPE_TEXTURE && (!allowCubemaps() || texture->GetType() != D3DRTYPE_CUBETEXTURE)) {
          continue;
        }

        const XXH64_hash_t texHash = texture->GetSampleView(true)->image()->getHash();

        // Currently we only support regular textures, skip lightmaps.
        if (lookupHash(RtxOptions::lightmapTextures(), texHash)) {
          continue;
        }

        // Allow for two stage candidates per texcoord index
        const uint32_t texcoordIndex = data[DXVK_TSS_TEXCOORDINDEX] & 0b111;
        const uint32_t candidateIndex = texcoordIndex * LegacyMaterialData::kMaxSupportedTextures;
        const uint32_t subIndex = (texcoordIndexToStage[candidateIndex] == kInvalidStage) ? 0 : 1;

        // Don't override if candidate exists
        if (texcoordIndexToStage[candidateIndex + subIndex] == kInvalidStage)
          texcoordIndexToStage[candidateIndex + subIndex] = stage;

        // Check if texture factor blending is enabled for the first stage
        if (useStageTextureFactorBlending && stage == 0) {
          isCurrentStageTextureFactorBlendingEnabled = isTextureFactorBlendingEnabled(d3d9State().textureStages[stage]);
        }

        // Check if texture factor blending is enabled
        if (isCurrentStageTextureFactorBlendingEnabled) {
          // Allow texture factor blending for textures explicitly in the allowBakedLightingTextures list
          const bool isExplicitlyAllowed = lookupHash(RtxOptions::allowBakedLightingTextures(), texHash);
          // Also allow texture factor blending for decal textures since they need proper blending
          const bool isDecalTexture = lookupHash(RtxOptions::decalTextures(), texHash) ||
                                      lookupHash(RtxOptions::dynamicDecalTextures(), texHash) ||
                                      lookupHash(RtxOptions::singleOffsetDecalTextures(), texHash) ||
                                      lookupHash(RtxOptions::nonOffsetDecalTextures(), texHash);
          // Also allow texture factor blending for particle textures since they need proper blending
          const bool isParticleTexture = lookupHash(RtxOptions::particleTextures(), texHash);
          if (!(isExplicitlyAllowed || isDecalTexture || isParticleTexture)) {
            useStageTextureFactorBlending = false;
            useMultipleStageTextureFactorBlending = false;
          }
        }
      }
    }

    // Find the ideal textures for raytracing, initialize the data to invalid (out of range) to unbind unused textures
    uint32_t firstStage = 0;

    // RENDER TARGET REPLACEMENT: Check if slot 0 contains a render target and find alternative texture
    int recommendedAlbedoSampler = -1;

    // AUTO-DETECT: Query decompiler for the correct albedo sampler binding
    if constexpr (!FixedFunction) {
      if (d3d9State().pixelShader.ptr() != nullptr) {
        const auto* commonShader = d3d9State().pixelShader->GetCommonShader();
        if (commonShader != nullptr) {
          const auto& bytecode = commonShader->GetBytecode();
          if (!bytecode.empty()) {
            const uint64_t psHash = XXH64(bytecode.data(), bytecode.size(), 0);
            if (auto* compatManager = m_shaderCompatibilityManager.get()) {
              int decompilerSampler = compatManager->getRecommendedAlbedoSampler(psHash);
              if (decompilerSampler >= 0) {
                recommendedAlbedoSampler = decompilerSampler;
                static int logCount = 0;
                if (++logCount <= 20) {
                  Logger::info(str::format("[RTX-AutoDetect] PS hash=0x", std::hex, psHash, std::dec,
                                          " decompiler says albedo is at slot ", decompilerSampler));
                }
                // AUTO-DETECT RENDER TARGET: If albedo is NOT at slot 0, and slot 0 has a texture,
                // then slot 0's texture is likely a render target
                if (decompilerSampler > 0 && d3d9State().textures[0] != nullptr) {
                  D3D9CommonTexture* tex0 = GetCommonTexture(d3d9State().textures[0]);
                  if (tex0 != nullptr) {
                    const XXH64_hash_t tex0Hash = tex0->GetImage()->getHash();
                    compatManager->registerDetectedRenderTarget(tex0Hash);
                  }
                }
              }
            }
          }
        }
      }
    }

    if constexpr (!FixedFunction) {
      auto isCategorizedRenderTarget = [this](const D3D9CommonTexture* texture) {
        if (texture == nullptr)
          return false;
        const auto& manualRenderTargets = RtxOptions::renderTargetReplacementTextures();

        auto matchesManualTag = [&](XXH64_hash_t hash) -> bool {
          return hash != 0 && lookupHash(manualRenderTargets, hash);
        };

        const XXH64_hash_t descriptorHash = texture->GetImage()->getDescriptorHash();
        if (matchesManualTag(descriptorHash)) {
          return true;
        }

        const XXH64_hash_t imageHash = texture->GetImage()->getHash();
        if (matchesManualTag(imageHash)) {
          return true;
        }

        // Check auto-detected render targets from decompiler analysis
        if (auto* compatManager = m_shaderCompatibilityManager.get()) {
          if (compatManager->isDetectedRenderTarget(imageHash)) {
            return true;
          }
        }

        if (const Rc<DxvkImageView> sampleView = texture->GetSampleView(true); sampleView != nullptr) {
          const Rc<DxvkImage>& viewImage = sampleView->image();
          if (viewImage != nullptr) {
            if (matchesManualTag(viewImage->getDescriptorHash()) || matchesManualTag(viewImage->getHash())) {
              return true;
            }
          }
        }

        return false;
      };

      // If slot 0 is categorized as render target, find alternative texture in other slots
      if (d3d9State().textures[0] != nullptr) {
        D3D9CommonTexture* tex0 = GetCommonTexture(d3d9State().textures[0]);
        const bool isRT = isCategorizedRenderTarget(tex0);
        // DEBUG: Log categorization check
        static int logCount = 0;
        if (++logCount <= 10 || (tex0 && tex0->GetImage()->getHash() == 0xc6702a5dbb13300c)) {
          Logger::info(str::format("[RTX-RT-Check] tex0Hash=0x", std::hex, (tex0 ? tex0->GetImage()->getHash() : 0), std::dec,
                                  " isRT=", isRT ? "YES" : "NO"));
        }
        if (tex0 && (tex0->GetType() == D3DRTYPE_TEXTURE || tex0->GetType() == D3DRTYPE_CUBETEXTURE) && isRT) {
          // Store the original render target hash FIRST, before searching for replacements
          // This is needed even when no replacement is found, for render target feedback detection
          m_activeDrawCallState.originalRenderTargetHash = tex0->GetImage()->getHash();

          // Try slots in order: s7, s8, s15, s5, s3, s2, s1, s6
          const int candidateSlots[] = {7, 8, 15, 5, 3, 2, 1, 6};
          for (int slot : candidateSlots) {
            if (d3d9State().textures[slot] != nullptr) {
              D3D9CommonTexture* tex = GetCommonTexture(d3d9State().textures[slot]);
              if (tex && (tex->GetType() == D3DRTYPE_TEXTURE || tex->GetType() == D3DRTYPE_CUBETEXTURE)) {
                const auto desc = tex->Desc();
                const bool isDepthStencil = (desc->Usage & D3DUSAGE_DEPTHSTENCIL) != 0;
                const bool isTinyTexture = (desc->Width <= 1 && desc->Height <= 1);
                const bool isRT = isCategorizedRenderTarget(tex);
                const XXH64_hash_t texHash = tex->GetImage()->getHash();

                // CRITICAL: Skip screen-sized textures - they're likely render targets even if not categorized
                const uint32_t vpW = static_cast<uint32_t>(m_activeDrawCallState.originalViewport.width);
                const uint32_t vpH = static_cast<uint32_t>(m_activeDrawCallState.originalViewport.height);
                const bool isScreenSized = (desc->Width == vpW && desc->Height == vpH);

                if (!isRT && !isDepthStencil && !isTinyTexture && !isScreenSized) {
                  // Only override if decompiler didn't provide an answer
                  if (recommendedAlbedoSampler < 0) {
                    recommendedAlbedoSampler = slot;
                  }
                  // CRITICAL: Set renderTargetReplacementSlot so shader capturer knows this is a RT replacement
                  m_activeDrawCallState.renderTargetReplacementSlot = slot;

                  break;
                }
              }
            }
          }
        }
      }
    }

    for (uint32_t idx = 0, textureID = 0; idx < NumTexcoordBins && textureID < LegacyMaterialData::kMaxSupportedTextures; idx++) {
      uint8_t stage;
      if constexpr (FixedFunction) {
        stage = texcoordIndexToStage[idx];
      } else {
        // Use recommended albedo sampler for first texture if render target was detected
        if (textureID == 0 && recommendedAlbedoSampler >= 0) {
          stage = recommendedAlbedoSampler;
        } else {
          stage = idx;
          if (recommendedAlbedoSampler >= 0 && stage == recommendedAlbedoSampler) {
            continue;  // Skip already-used albedo sampler
          }
        }
      }
      if (stage == kInvalidStage || d3d9State().textures[stage] == nullptr)
        continue;

      D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
      assert(pTexInfo != nullptr);

      // Log texture being added to material
      static uint32_t texAddLogCount = 0;
      if (++texAddLogCount <= 100) {
        Logger::info(str::format("[TEX-ADD] textureID=", textureID, " stage=", stage,
                                " hash=0x", std::hex, pTexInfo->GetImage()->getHash(), std::dec,
                                " size=", pTexInfo->Desc()->Width, "x", pTexInfo->Desc()->Height));
      }

      // Send the texture stage state for first texture slot (or 0th stage if no texture)
      if (textureID == 0) {
        // ColorTexture2 is optional and currently only used as RayPortal material, the material type will be checked in the submitDrawState.
        // So we don't use it to check valid drawcall or not here.
        if (pTexInfo->GetImage()->getHash() == kEmptyHash) {
          // Allow draws with empty texture hash if shader output capture is enabled
          // Shader capture will generate textures from pixel shader output for these draws
          const bool allowForShaderCapture = ShaderOutputCapturer::enableShaderOutputCapture() &&
                                             (ShaderOutputCapturer::captureAllDraws() ||
                                              !ShaderOutputCapturer::captureEnabledHashes().empty());
          if (!allowForShaderCapture) {
            ONCE(Logger::info("[RTX-Compatibility-Info] Texture 0 without valid hash detected, skipping drawcall."));
            return false;
          } else {
            Logger::info("[RTX-Compatibility-Info] Texture 0 without valid hash - allowing for shader output capture");
          }
        }

        if (FixedFunction) {
          firstStage = stage;
        }
      }

      D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
      XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);

      Rc<DxvkSampler> sampler;
      auto samplerIt = m_samplerCache.find(samplerHash);
      if (samplerIt != m_samplerCache.end()) {
        sampler = samplerIt->second;
      } else {
        const auto samplerInfo = m_parent->DecodeSamplerKey(key);
        sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
        m_samplerCache.insert(std::make_pair(samplerHash, sampler));
      }

      // Cache the slot we want to bind
      const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
      m_activeDrawCallState.materialData.colorTextures[textureID] = TextureRef(pTexInfo->GetSampleView(srgb));
      m_activeDrawCallState.materialData.samplers[textureID] = sampler;

      auto shaderSampler = RemapStateSamplerShader(stage);
      m_activeDrawCallState.materialData.colorTextureSlot[textureID] = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));

      // SHADER CAPTURE: Capture the D3D9 texture binding for shader re-execution
      // This preserves the ORIGINAL game textures before RT replacement happens
      DrawCallState::CapturedD3D9Texture capturedTex;
      capturedTex.texture = TextureRef(pTexInfo->GetSampleView(srgb));
      capturedTex.sampler = sampler;  // Capture sampler state
      capturedTex.slot = m_activeDrawCallState.materialData.colorTextureSlot[textureID];
      m_activeDrawCallState.capturedD3D9Textures.push_back(capturedTex);

      // DEBUG: Log first texture capture to see if remapping worked
      if (textureID == 0) {
        const XXH64_hash_t capturedHash = pTexInfo->GetImage()->getHash();
        if (capturedHash == 0xc6702a5dbb13300c) {
          Logger::warn(str::format("[RTX-TextureCapture-PROBLEM] textureID=0 stage=", stage,
                                  " textureHash=0x", std::hex, capturedHash, std::dec,
                                  " recommendedAlbedoSampler=", recommendedAlbedoSampler,
                                  " WARNING: Captured the RENDER TARGET itself!"));
        }
      }

      ++textureID;
    }

    // SHADER CAPTURE FIX: Capture ALL remaining D3D9 texture stages that weren't included in material
    // The pixel shader might reference texture stages beyond kMaxSupportedTextures
    if constexpr (!FixedFunction) {
      for (uint32_t stage = 0; stage < caps::MaxTexturesPS; stage++) {
        if (d3d9State().textures[stage] == nullptr)
          continue;

        // Check if we already captured this stage
        bool alreadyCaptured = false;
        for (const auto& captured : m_activeDrawCallState.capturedD3D9Textures) {
          auto shaderSampler = RemapStateSamplerShader(stage);
          uint32_t stageSlot = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));
          if (captured.slot == stageSlot) {
            alreadyCaptured = true;
            break;
          }
        }

        if (!alreadyCaptured) {
          D3D9CommonTexture* pTexInfo = GetCommonTexture(d3d9State().textures[stage]);
          if (pTexInfo && (pTexInfo->GetType() == D3DRTYPE_TEXTURE || pTexInfo->GetType() == D3DRTYPE_CUBETEXTURE)) {
            // Skip depth/stencil textures - they produce black when sampled as color
            const VkFormat texFormat = pTexInfo->GetImage()->info().format;
            const bool isDepthFormat = (texFormat == VK_FORMAT_D16_UNORM ||
                                       texFormat == VK_FORMAT_D24_UNORM_S8_UINT ||
                                       texFormat == VK_FORMAT_D32_SFLOAT ||
                                       texFormat == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                                       texFormat == VK_FORMAT_D16_UNORM_S8_UINT);
            if (isDepthFormat) {
              continue;  // Skip depth textures for shader capture
            }

            const bool srgb = d3d9State().samplerStates[stage][D3DSAMP_SRGBTEXTURE] & 0x1;
            auto shaderSampler = RemapStateSamplerShader(stage);
            uint32_t stageSlot = computeResourceSlotId(shaderSampler.first, DxsoBindingType::Image, uint32_t(shaderSampler.second));

            // Create/get sampler for this stage
            D3D9SamplerKey key = m_parent->CreateSamplerKey(stage);
            XXH64_hash_t samplerHash = D3D9SamplerKeyHash{}(key);
            Rc<DxvkSampler> sampler;
            auto samplerIt = m_samplerCache.find(samplerHash);
            if (samplerIt != m_samplerCache.end()) {
              sampler = samplerIt->second;
            } else {
              const auto samplerInfo = m_parent->DecodeSamplerKey(key);
              sampler = m_parent->GetDXVKDevice()->createSampler(samplerInfo);
              m_samplerCache.insert(std::make_pair(samplerHash, sampler));
            }

            DrawCallState::CapturedD3D9Texture capturedTex;
            capturedTex.texture = TextureRef(pTexInfo->GetSampleView(srgb));
            capturedTex.sampler = sampler;  // Capture sampler state
            capturedTex.slot = stageSlot;
            m_activeDrawCallState.capturedD3D9Textures.push_back(capturedTex);
          }
        }
      }
    }

    // Update the drawcall state with texture stage info
    setTextureStageState(d3d9State(), firstStage, useStageTextureFactorBlending, useMultipleStageTextureFactorBlending,
                         m_activeDrawCallState.materialData, m_activeDrawCallState.transformData);

    if (d3d9State().textures[firstStage]) {
      m_activeDrawCallState.setupCategoriesForTexture();

      // Check if an ignore texture is bound
      if (m_activeDrawCallState.getCategoryFlags().test(InstanceCategories::Ignore)) {
        return false;
      }

      if (m_activeDrawCallState.testCategoryFlags(InstanceCategories::Terrain)) {
        if (RtxOptions::terrainAsDecalsEnabledIfNoBaker() && !TerrainBaker::enableBaking()) {

          m_activeDrawCallState.removeCategory(InstanceCategories::Terrain);
          m_activeDrawCallState.setCategory(InstanceCategories::DecalStatic, true);

          // modulate to compensate the multilayer blending
          DxvkRtTextureOperation& texop = m_activeDrawCallState.materialData.textureColorOperation;
          if (RtxOptions::terrainAsDecalsAllowOverModulate()) {
            if (texop == DxvkRtTextureOperation::Modulate2x || texop == DxvkRtTextureOperation::Modulate4x) {
              texop = DxvkRtTextureOperation::Force_Modulate2x;
            }
          }
        }
      }

      if (!m_forceGeometryCopy && RtxOptions::alwaysCopyDecalGeometries()) {
        // Only poke decal hashes when option is enabled.
        m_forceGeometryCopy |= m_activeDrawCallState.testCategoryFlags(CATEGORIES_REQUIRE_GEOMETRY_COPY);
      }
    }

    m_texcoordIndex = d3d9State().textureStages[firstStage][DXVK_TSS_TEXCOORDINDEX];

    // Log final texture summary for this draw call
    static uint32_t texSummaryCount = 0;
    if (++texSummaryCount <= 100) {
      Logger::info(str::format("[TEX-SUMMARY] DrawCall completed: firstStage=", firstStage,
                              " renderTargetReplacementSlot=", m_activeDrawCallState.renderTargetReplacementSlot,
                              " originalRT=0x", std::hex, m_activeDrawCallState.originalRenderTargetHash, std::dec));
    }

    return true;
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context) {
    if (!RtxOptions::enableRaytracing() || !m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_parent->PrepareTextures();

    IndexContext indices;
    if (indexed) {
      D3D9CommonBuffer* ibo = GetCommonBuffer(d3d9State().indices);
      assert(ibo != nullptr);

      indices.ibo = ibo;
      indices.indexBuffer = ibo->GetMappedSlice();
      indices.indexType = DecodeIndexType(ibo->Desc()->Format);
    }

    // Copy over the vertex buffers that are actually required
    VertexContext vertices[caps::MaxStreams];
    for (uint32_t i = 0; i < caps::MaxStreams; i++) {
      const auto& dx9Vbo = d3d9State().vertexBuffers[i];
      auto* vbo = GetCommonBuffer(dx9Vbo.vertexBuffer);
      if (vbo != nullptr) {
        vertices[i].stride = dx9Vbo.stride;
        vertices[i].offset = dx9Vbo.offset;
        vertices[i].buffer = vbo->GetBufferSlice<D3D9_COMMON_BUFFER_TYPE_MAPPING>();
        vertices[i].mappedSlice = vbo->GetMappedSlice();
        vertices[i].pVBO = vbo;

        // If staging upload has been enabled on a buffer then previous buffer lock:
        //   a) triggered a pipeline stall (overlapped mapped ranges, improper flags etc)
        //   b) does not have D3DLOCK_DONOTWAIT, or was in use at Map()
        // 
        // Buffers with staged uploads may have contents valid ONLY until next Map().
        // We must NOT use such buffer directly and have to always copy the contents.
        vertices[i].canUseBuffer = vbo->DoesStagingBufferUploads() == false;
      }
    }

    return internalPrepareDraw(indices, vertices, context);
  }

  PrepareDrawFlags D3D9Rtx::PrepareDrawUPGeometryForRT(const bool indexed,
                                                       const D3D9BufferSlice& buffer,
                                                       const D3DFORMAT indexFormat,
                                                       const uint32_t indexSize,
                                                       const uint32_t indexOffset,
                                                       const uint32_t vertexSize,
                                                       const uint32_t vertexStride,
                                                       const DrawContext& drawContext) {
    if (!RtxOptions::enableRaytracing() || !m_enableDrawCallConversion) {
      return PrepareDrawFlag::PreserveDrawCallAndItsState;
    }

    m_parent->PrepareTextures();

    // 'buffer' - contains vertex + index data (packed in that order)

    IndexContext indices;
    if (indexed) {
      indices.indexBuffer = buffer.slice.getSliceHandle(indexOffset, indexSize);
      indices.indexType = DecodeIndexType(static_cast<D3D9Format>(indexFormat));
    }

    VertexContext vertices[caps::MaxStreams];
    vertices[0].stride = vertexStride;
    vertices[0].offset = 0;
    vertices[0].buffer = buffer.slice.subSlice(0, vertexSize);
    vertices[0].mappedSlice = buffer.slice.getSliceHandle(0, vertexSize);
    vertices[0].canUseBuffer = true;

    return internalPrepareDraw(indices, vertices, drawContext);
  }

  void D3D9Rtx::ResetSwapChain(const D3DPRESENT_PARAMETERS& presentationParameters) {
    // Early out if the cached present parameters are not out of date

    if (m_activePresentParams.has_value()) {
      if (
        m_activePresentParams->BackBufferWidth == presentationParameters.BackBufferWidth &&
        m_activePresentParams->BackBufferHeight == presentationParameters.BackBufferHeight &&
        m_activePresentParams->BackBufferFormat == presentationParameters.BackBufferFormat &&
        m_activePresentParams->BackBufferCount == presentationParameters.BackBufferCount &&
        m_activePresentParams->MultiSampleType == presentationParameters.MultiSampleType &&
        m_activePresentParams->MultiSampleQuality == presentationParameters.MultiSampleQuality &&
        m_activePresentParams->SwapEffect == presentationParameters.SwapEffect &&
        m_activePresentParams->hDeviceWindow == presentationParameters.hDeviceWindow &&
        m_activePresentParams->Windowed == presentationParameters.Windowed &&
        m_activePresentParams->EnableAutoDepthStencil == presentationParameters.EnableAutoDepthStencil &&
        m_activePresentParams->AutoDepthStencilFormat == presentationParameters.AutoDepthStencilFormat &&
        m_activePresentParams->Flags == presentationParameters.Flags &&
        m_activePresentParams->FullScreen_RefreshRateInHz == presentationParameters.FullScreen_RefreshRateInHz &&
        m_activePresentParams->PresentationInterval == presentationParameters.PresentationInterval
      ) {
        return;
      }
    }

    // Cache the present parameters
    m_activePresentParams = presentationParameters;

    // Inform the backend about potential presenter update
    m_parent->EmitCs([cWidth = m_activePresentParams->BackBufferWidth,
                      cHeight = m_activePresentParams->BackBufferHeight](DxvkContext* ctx) {
      static_cast<RtxContext*>(ctx)->resetScreenResolution({ cWidth, cHeight , 1 });
    });
  }

  void D3D9Rtx::EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx) {
    const auto currentReflexFrameId = GetReflexFrameId();
    
    // Flush any pending game and RTX work
    m_parent->Flush();

    // Inform backend of end-frame
    m_parent->EmitCs([currentReflexFrameId, targetImage, callInjectRtx](DxvkContext* ctx) { 
      static_cast<RtxContext*>(ctx)->endFrame(currentReflexFrameId, targetImage, callInjectRtx); 
    });

    // Reset for the next frame
    m_rtxInjectTriggered = false;
    m_drawCallID = 0;
    m_seenCameraPositionsPrev = std::move(m_seenCameraPositions);

    m_stagedBonesCount = 0;
  }

  void D3D9Rtx::OnPresent(const Rc<DxvkImage>& targetImage) {
    // NV-DXVK start: Debug logging for presented target
    Logger::info(str::format("[RTX-PRESENT] Presenting to target: format=", targetImage->info().format,
      " size=", targetImage->info().extent.width, "x", targetImage->info().extent.height,
      " usage=", targetImage->info().usage,
      " tiling=", targetImage->info().tiling));
    // NV-DXVK end

    // Inform backend of present
    m_parent->EmitCs([targetImage](DxvkContext* ctx) { static_cast<RtxContext*>(ctx)->onPresent(targetImage); });
  }

  void D3D9Rtx::captureOriginalD3D9Buffers(const Direct3DState9& state) {
    // DEBUG: Log timing to compare with SetVertexShaderConstantF calls
    static uint32_t captureTimingLogCount = 0;
    if (++captureTimingLogCount <= 10) {
      Logger::info(str::format("[CAPTURE-TIMING] #", captureTimingLogCount, " captureOriginalD3D9Buffers CALLED - starting capture"));
    }

    // Clear previous captures
    m_activeDrawCallState.capturedVertexStreams.clear();
    m_activeDrawCallState.capturedVertexElements.clear();

    // Capture D3D9 shader objects for re-execution (extract raw pointers from Com smart pointers)
    m_activeDrawCallState.vertexShader = state.vertexShader.ptr();
    m_activeDrawCallState.pixelShader = state.pixelShader.ptr();
    m_activeDrawCallState.vertexDecl = state.vertexDecl.ptr();

    // Capture shader constants
    // Vertex shader constants (D3D9 supports up to 256 float4 constants for VS software, 256 for hardware)
    if (state.vertexShader != nullptr) {
      const uint32_t vsConstantCount = caps::MaxFloatConstantsVS; // 256 constants
      m_activeDrawCallState.vertexShaderConstantData.resize(vsConstantCount);

      // DEBUG: Scan ALL VS constants to find non-zero values (game may use unexpected registers)
      static uint32_t vsConstLogCount = 0;
      if (++vsConstLogCount <= 5) {
        Logger::info(str::format("[VS-CONST-CAPTURE] #", vsConstLogCount, " Scanning ALL ", vsConstantCount, " VS constants for non-zero values:"));
        uint32_t nonZeroCount = 0;
        for (uint32_t i = 0; i < vsConstantCount; i++) {
          const auto& c = state.vsConsts.fConsts[i];
          if (c.x != 0.0f || c.y != 0.0f || c.z != 0.0f || c.w != 0.0f) {
            nonZeroCount++;
            Logger::info(str::format("[VS-CONST-CAPTURE]   c[", i, "] = (", c.x, ", ", c.y, ", ", c.z, ", ", c.w, ") <-- NON-ZERO!"));
          }
        }
        if (nonZeroCount == 0) {
          Logger::warn("[VS-CONST-CAPTURE]   ALL 256 VS CONSTANTS ARE ZERO! Game may use fixed-function transforms.");
        } else {
          Logger::info(str::format("[VS-CONST-CAPTURE]   Found ", nonZeroCount, " non-zero constants"));
        }
      }

      // Copy from fConsts array (float constants)
      memcpy(m_activeDrawCallState.vertexShaderConstantData.data(), state.vsConsts.fConsts, vsConstantCount * sizeof(Vector4));

      // CRITICAL FIX: If VS constants c[0]-c[3] are all zeros but fixed-function transforms are set,
      // inject the WVP matrix into c[0]-c[3]. Many D3D9 games rely on the driver to do this.
      // Check if c[0]-c[3] are all zeros (transformation matrix would be zero)
      bool c0to3AllZeros = true;
      for (int i = 0; i < 4 && c0to3AllZeros; i++) {
        const auto& c = state.vsConsts.fConsts[i];
        if (c.x != 0.0f || c.y != 0.0f || c.z != 0.0f || c.w != 0.0f) {
          c0to3AllZeros = false;
        }
      }

      if (c0to3AllZeros) {
        // Get fixed-function transforms
        const Matrix4& world = d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)];
        const Matrix4& view = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
        const Matrix4& proj = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

        // Check if fixed-function transforms are valid (not identity or zero)
        bool hasValidTransforms = proj[0][0] != 0.0f || proj[1][1] != 0.0f || proj[2][2] != 0.0f;

        if (hasValidTransforms) {
          // Compute WVP = World * View * Projection (row-major for D3D9)
          Matrix4 wvp = world * view * proj;

          // Inject WVP matrix into c[0]-c[3] (each row is one constant register)
          // D3D9 uses row-major matrices, so row 0 goes to c[0], etc.
          for (int row = 0; row < 4; row++) {
            m_activeDrawCallState.vertexShaderConstantData[row].x = wvp[row][0];
            m_activeDrawCallState.vertexShaderConstantData[row].y = wvp[row][1];
            m_activeDrawCallState.vertexShaderConstantData[row].z = wvp[row][2];
            m_activeDrawCallState.vertexShaderConstantData[row].w = wvp[row][3];
          }

          static uint32_t wvpInjectLogCount = 0;
          if (++wvpInjectLogCount <= 10) {
            Logger::info(str::format("[VS-CONST-INJECT] #", wvpInjectLogCount,
              " Injected WVP matrix into c[0]-c[3] (was all zeros)"));
            Logger::info(str::format("[VS-CONST-INJECT]   c[0] = (",
              m_activeDrawCallState.vertexShaderConstantData[0].x, ", ",
              m_activeDrawCallState.vertexShaderConstantData[0].y, ", ",
              m_activeDrawCallState.vertexShaderConstantData[0].z, ", ",
              m_activeDrawCallState.vertexShaderConstantData[0].w, ")"));
          }
        }
      }

      // DEBUG: Also log fixed-function transforms for comparison
      if (vsConstLogCount <= 5) {
        const Matrix4& world = d3d9State().transforms[GetTransformIndex(D3DTS_WORLD)];
        const Matrix4& view = d3d9State().transforms[GetTransformIndex(D3DTS_VIEW)];
        const Matrix4& proj = d3d9State().transforms[GetTransformIndex(D3DTS_PROJECTION)];

        bool worldNonZero = world[0][0] != 0.0f || world[1][1] != 0.0f || world[2][2] != 0.0f;
        bool viewNonZero = view[0][0] != 0.0f || view[1][1] != 0.0f || view[2][2] != 0.0f;
        bool projNonZero = proj[0][0] != 0.0f || proj[1][1] != 0.0f || proj[2][2] != 0.0f;

        Logger::info(str::format("[VS-CONST-CAPTURE] Fixed-function transforms:"));
        Logger::info(str::format("[VS-CONST-CAPTURE]   D3DTS_WORLD non-zero: ", worldNonZero ? "YES" : "NO",
                                " diagonal=(", world[0][0], ",", world[1][1], ",", world[2][2], ")"));
        Logger::info(str::format("[VS-CONST-CAPTURE]   D3DTS_VIEW non-zero: ", viewNonZero ? "YES" : "NO",
                                " diagonal=(", view[0][0], ",", view[1][1], ",", view[2][2], ")"));
        Logger::info(str::format("[VS-CONST-CAPTURE]   D3DTS_PROJECTION non-zero: ", projNonZero ? "YES" : "NO",
                                " diagonal=(", proj[0][0], ",", proj[1][1], ",", proj[2][2], ")"));
      }

    } else {
      m_activeDrawCallState.vertexShaderConstantData.clear();
    }

    // Pixel shader constants (D3D9 supports up to 224 float4 constants)
    if (state.pixelShader != nullptr) {
      const uint32_t psConstantCount = caps::MaxFloatConstantsPS; // 224 constants
      m_activeDrawCallState.pixelShaderConstantData.resize(psConstantCount);

      // TARGETED DEBUG: Log PS constants BEFORE copy - check c[30], c[150], c[180] where game sets values
      static uint32_t psConstLogCount = 0;
      if (++psConstLogCount <= 5) {
        Logger::info(str::format("[PS-CONST-CAPTURE] #", psConstLogCount, " BEFORE memcpy:"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   state.psConsts.fConsts[30] = (",
                                state.psConsts.fConsts[30].x, ", ", state.psConsts.fConsts[30].y, ", ",
                                state.psConsts.fConsts[30].z, ", ", state.psConsts.fConsts[30].w, ")"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   state.psConsts.fConsts[150] = (",
                                state.psConsts.fConsts[150].x, ", ", state.psConsts.fConsts[150].y, ", ",
                                state.psConsts.fConsts[150].z, ", ", state.psConsts.fConsts[150].w, ")"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   state.psConsts.fConsts[180] = (",
                                state.psConsts.fConsts[180].x, ", ", state.psConsts.fConsts[180].y, ", ",
                                state.psConsts.fConsts[180].z, ", ", state.psConsts.fConsts[180].w, ")"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   psConstantCount=", psConstantCount));
      }

      // Copy from fConsts array (float constants)
      memcpy(m_activeDrawCallState.pixelShaderConstantData.data(), state.psConsts.fConsts, psConstantCount * sizeof(Vector4));

      // TARGETED DEBUG: Log PS constants AFTER copy
      if (psConstLogCount <= 5) {
        Logger::info(str::format("[PS-CONST-CAPTURE] #", psConstLogCount, " AFTER memcpy:"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   pixelShaderConstantData[30] = (",
                                m_activeDrawCallState.pixelShaderConstantData[30].x, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[30].y, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[30].z, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[30].w, ")"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   pixelShaderConstantData[150] = (",
                                m_activeDrawCallState.pixelShaderConstantData[150].x, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[150].y, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[150].z, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[150].w, ")"));
        Logger::info(str::format("[PS-CONST-CAPTURE]   pixelShaderConstantData[180] = (",
                                m_activeDrawCallState.pixelShaderConstantData[180].x, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[180].y, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[180].z, ", ",
                                m_activeDrawCallState.pixelShaderConstantData[180].w, ")"));
      }
      // SHADER-BASED LIGHT INJECTION
      // LEGO Batman 2 shaders expect lighting in c32-35 but game never calls SetLight
      // Detect these constants and inject RTX lights automatically
      static uint64_t shaderLightFrameCounter = 0;
      static uint64_t lastInjectionFrame = UINT64_MAX;
      shaderLightFrameCounter++;

      // Reset injection flag every 1000 draw calls (approximate frame boundary)
      if (shaderLightFrameCounter - lastInjectionFrame > 1000) {
        m_shaderLightsInjectedThisFrame = false;
      }

      if (!m_shaderLightsInjectedThisFrame && m_activeDrawCallState.pixelShaderConstantData.size() > 35) {
        const Vector4& lightColor0 = m_activeDrawCallState.pixelShaderConstantData[32];
        const Vector4& lightColor1 = m_activeDrawCallState.pixelShaderConstantData[33];
        const Vector4& lightDir0 = m_activeDrawCallState.pixelShaderConstantData[34];
        const Vector4& lightDir1 = m_activeDrawCallState.pixelShaderConstantData[35];

        auto isZero = [](const Vector4& v) {
          return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
        };

        // Check if game has NO D3D9 lights enabled
        bool hasD3D9Lights = false;
        for (auto idx : d3d9State().enabledLightIndices) {
          if (idx != UINT32_MAX) {
            hasD3D9Lights = true;
            break;
          }
        }

        // If no D3D9 lights and shader expects lighting, inject synthetic lights
        if (!hasD3D9Lights) {
          std::vector<D3DLIGHT9> syntheticLights;

          // Create light 0 - main directional light
          D3DLIGHT9 light0 = {};
          light0.Type = D3DLIGHT_DIRECTIONAL;
          if (!isZero(lightColor0) && !isZero(lightDir0)) {
            // Use shader constants if available
            light0.Diffuse = { lightColor0.x, lightColor0.y, lightColor0.z, lightColor0.w };
            light0.Direction = { lightDir0.x, lightDir0.y, lightDir0.z };
            Logger::info(str::format("[SHADER-LIGHT] Using shader constants for light0: color=(",
              lightColor0.x, ",", lightColor0.y, ",", lightColor0.z, ") dir=(",
              lightDir0.x, ",", lightDir0.y, ",", lightDir0.z, ")"));
          } else {
            // Default sun-like light from above-front
            light0.Diffuse = { 1.0f, 0.95f, 0.9f, 1.0f };
            light0.Direction = { -0.3f, -0.8f, 0.5f };
            Logger::info("[SHADER-LIGHT] Injecting default light0 (sun-like directional)");
          }
          light0.Ambient = { 0.1f, 0.1f, 0.15f, 1.0f };
          light0.Specular = { 0.8f, 0.8f, 0.8f, 1.0f };
          syntheticLights.push_back(light0);

          // Create light 1 - fill light
          D3DLIGHT9 light1 = {};
          light1.Type = D3DLIGHT_DIRECTIONAL;
          if (!isZero(lightColor1) && !isZero(lightDir1)) {
            light1.Diffuse = { lightColor1.x, lightColor1.y, lightColor1.z, lightColor1.w };
            light1.Direction = { lightDir1.x, lightDir1.y, lightDir1.z };
            Logger::info(str::format("[SHADER-LIGHT] Using shader constants for light1: color=(",
              lightColor1.x, ",", lightColor1.y, ",", lightColor1.z, ") dir=(",
              lightDir1.x, ",", lightDir1.y, ",", lightDir1.z, ")"));
          } else {
            // Default fill light from the side
            light1.Diffuse = { 0.3f, 0.35f, 0.4f, 1.0f };
            light1.Direction = { 0.7f, -0.2f, -0.6f };
            Logger::info("[SHADER-LIGHT] Injecting default light1 (fill directional)");
          }
          light1.Ambient = { 0.05f, 0.05f, 0.08f, 1.0f };
          light1.Specular = { 0.3f, 0.3f, 0.3f, 1.0f };
          syntheticLights.push_back(light1);

          // Add the lights to RTX
          if (!syntheticLights.empty()) {
            m_parent->EmitCs([syntheticLights](DxvkContext* ctx) {
              static_cast<RtxContext*>(ctx)->addLights(syntheticLights.data(), syntheticLights.size());
            });
            m_shaderLightsInjectedThisFrame = true;
            lastInjectionFrame = shaderLightFrameCounter;
            Logger::info(str::format("[SHADER-LIGHT] Injected ", syntheticLights.size(), " synthetic lights at draw #", shaderLightFrameCounter));
          }
        } else {
          // Log why we didn't inject (only first few times)
          static uint32_t noInjectLogCount = 0;
          if (noInjectLogCount++ < 5) {
            Logger::info(str::format("[SHADER-LIGHT] Not injecting: hasD3D9Lights=", hasD3D9Lights));
          }
        }
      }
    } else {
      m_activeDrawCallState.pixelShaderConstantData.clear();
      Logger::info("[PS-CONST-CAPTURE] No pixel shader - constants cleared");
    }

    // Bail if no vertex declaration
    if (state.vertexDecl == nullptr) {
      return;
    }

    // Get the vertex elements from the declaration
    const auto& elements = state.vertexDecl->GetElements();
    if (elements.empty()) {
      return;
    }

    // Track which streams we've already captured to avoid duplication
    std::unordered_map<uint32_t, uint32_t> capturedStreams;

    // Step 1: Copy the vertex element declarations
    for (const auto& elem : elements) {
      if (elem.Stream == 0xFF) break; // End marker

      DrawCallState::CapturedVertexElement capturedElem;
      capturedElem.stream = elem.Stream;
      capturedElem.offset = elem.Offset;
      capturedElem.type = elem.Type;
      capturedElem.method = elem.Method;
      capturedElem.usage = elem.Usage;
      capturedElem.usageIndex = elem.UsageIndex;
      m_activeDrawCallState.capturedVertexElements.push_back(capturedElem);

      // Check if we've already captured this stream
      if (capturedStreams.count(elem.Stream)) {
        continue;
      }

      // Get the vertex buffer state for this stream
      const auto& vbState = state.vertexBuffers[elem.Stream];
      if (vbState.vertexBuffer == nullptr || vbState.stride == 0) {
        continue;
      }

      // Get the common buffer
      auto* common = GetCommonBuffer(vbState.vertexBuffer);
      if (!common) {
        continue;
      }

      // Calculate offset and size
      VkDeviceSize offset = vbState.offset;
      if (m_activeDrawCallState.originalBaseVertex > 0) {
        offset += VkDeviceSize(m_activeDrawCallState.originalBaseVertex) * vbState.stride;
      }

      const uint32_t vertexCount = m_activeDrawCallState.geometryData.vertexCount;
      VkDeviceSize bytes = VkDeviceSize(vertexCount) * vbState.stride;

      // Validate bounds
      if (offset >= common->Desc()->Size || bytes == 0) {
        continue;
      }
      bytes = std::min(bytes, common->Desc()->Size - offset);

      // Lock the buffer for reading
      void* src = nullptr;
      if (FAILED(common->Lock(uint32_t(offset), uint32_t(bytes), &src,
                 D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK)) || !src) {
        if (src) common->Unlock();
        continue;
      }

      // Copy the data into our captured stream
      DrawCallState::CapturedVertexStream& captured =
        m_activeDrawCallState.capturedVertexStreams.emplace_back();
      captured.streamIndex = elem.Stream;
      captured.stride = vbState.stride;
      captured.data.assign(static_cast<uint8_t*>(src),
                           static_cast<uint8_t*>(src) + size_t(bytes));

      common->Unlock();
      capturedStreams.emplace(elem.Stream,
                             uint32_t(m_activeDrawCallState.capturedVertexStreams.size() - 1));

      // CRITICAL FIX: Set originalVertexStride from first captured stream (stream 0)
      // This is used by shader re-execution (Option C) to replay vertex buffers
      if (elem.Stream == 0 && m_activeDrawCallState.originalVertexStride == 0) {
        m_activeDrawCallState.originalVertexStride = captured.stride;
      }
    }

    // Capture index buffer if present AND the draw uses indices
    if (state.indices != nullptr && m_activeDrawCallState.geometryData.indexCount > 0) {
      auto* ibo = GetCommonBuffer(state.indices);
      if (ibo != nullptr) {
        // Get index type from the buffer description
        const D3D9Format iboFormat = ibo->Desc()->Format;
        const VkIndexType indexType = DecodeIndexType(iboFormat);
        const uint32_t indexStride = (indexType == VK_INDEX_TYPE_UINT16) ? 2 : 4;

        // Get index count from geometry data (already calculated by processIndexBuffer)
        const uint32_t indexCount = m_activeDrawCallState.geometryData.indexCount;

        // Calculate offset - we need to account for the start index from the draw call
        // Note: originalFirstIndex should be set before this function is called
        const uint32_t indexOffset = m_activeDrawCallState.originalFirstIndex * indexStride;
        const uint32_t bytes = indexCount * indexStride;

        // Validate bounds
        if (indexOffset + bytes <= ibo->Desc()->Size) {
          // Lock the index buffer for reading
          void* src = nullptr;
          if (SUCCEEDED(ibo->Lock(indexOffset, bytes, &src, D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK)) && src) {
            // Copy index data
            m_activeDrawCallState.originalIndexData.assign(
              static_cast<uint8_t*>(src),
              static_cast<uint8_t*>(src) + bytes);

            // Calculate min/max indices from captured data (needed for index rebasing)
            uint32_t minIndex = 0;
            uint32_t maxIndex = 0;
            if (indexType == VK_INDEX_TYPE_UINT16) {
              const uint16_t* indices = reinterpret_cast<const uint16_t*>(m_activeDrawCallState.originalIndexData.data());
              fast::findMinMax<uint16_t>(indexCount, const_cast<uint16_t*>(indices), minIndex, maxIndex);
            } else {
              const uint32_t* indices = reinterpret_cast<const uint32_t*>(m_activeDrawCallState.originalIndexData.data());
              fast::findMinMax<uint32_t>(indexCount, const_cast<uint32_t*>(indices), minIndex, maxIndex);
            }

            // Set all metadata fields required for shader re-execution
            m_activeDrawCallState.originalIndexMin = minIndex;
            m_activeDrawCallState.originalIndexCount = indexCount;
            m_activeDrawCallState.originalIndexType = indexType;
            m_activeDrawCallState.originalIndexOffset = indexOffset;

            ibo->Unlock();

          } else {
            if (src) ibo->Unlock();
          }
        }
      }
    }

    // DEBUG: Log capture result BEFORE validation
    static uint32_t captureResultLog = 0;
    const uint32_t finalVertCount = m_activeDrawCallState.geometryData.vertexCount;
    const size_t streamCount = m_activeDrawCallState.capturedVertexStreams.size();
    if (finalVertCount > 6 && captureResultLog++ < 50) {
      Logger::info(str::format("[CAPTURE-RESULT] verts=", finalVertCount, " capturedStreams=", streamCount,
        " BEFORE shouldCaptureStatic check"));
    }

    // CRITICAL FIX: Validate VS constants AFTER vertex buffer capture completes
    // If matrices are invalid (all zeros), clear ALL captured data to mark it as invalid
    // But DON'T return early - we've already done the capture work
    if (!ShaderOutputCapturer::shouldCaptureStatic(m_activeDrawCallState)) {
      if (finalVertCount > 6 && captureResultLog < 100) {
        Logger::warn(str::format("[CAPTURE-CLEARED] verts=", finalVertCount, " streamCount=", streamCount,
          " - shouldCaptureStatic returned FALSE, clearing captured data!"));
      }
      m_activeDrawCallState.vertexShaderConstantData.clear();
      m_activeDrawCallState.capturedVertexStreams.clear();
      m_activeDrawCallState.capturedVertexElements.clear();
      m_activeDrawCallState.originalIndexData.clear();
    } else {
      if (finalVertCount > 6 && captureResultLog < 100) {
        Logger::info(str::format("[CAPTURE-KEPT] verts=", finalVertCount, " streamCount=", streamCount,
          " - shouldCaptureStatic returned TRUE, keeping captured data"));
      }
    }
  }

  bool D3D9Rtx::shouldCaptureBuffers(const Direct3DState9& state) const {
    // OPTIMIZATION: Quick check before expensive buffer copies
    // Only copy buffers if this draw call will actually be captured for shader re-execution

    // DEBUG: Log all calls with geometry info
    static uint32_t shouldCaptureLog = 0;
    const uint32_t vertCount = m_activeDrawCallState.geometryData.vertexCount;
    const bool logThis = (vertCount > 6 && shouldCaptureLog++ < 50);

    // Requirement 1: Must have both vertex and pixel shaders
    if (state.vertexShader == nullptr || state.pixelShader == nullptr) {
      if (logThis) {
        Logger::info(str::format("[SHOULD-CAPTURE-BUFFERS] verts=", vertCount,
          " REJECTED: missing shader (VS=", state.vertexShader != nullptr,
          " PS=", state.pixelShader != nullptr, ")"));
      }
      return false;
    }

    // Requirement 2: Must have vertex declaration
    if (state.vertexDecl == nullptr) {
      if (logThis) {
        Logger::info(str::format("[SHOULD-CAPTURE-BUFFERS] verts=", vertCount, " REJECTED: no vertex declaration"));
      }
      return false;
    }

    // CRITICAL PERF FIX: Check if this specific material needs capture
    // If the material is already cached (static material), don't waste time capturing buffers!
    // This prevents hundreds of expensive buffer locks/copies per frame for cached materials.
    const bool needsCapture = ShaderOutputCapturer::shouldCaptureStatic(m_activeDrawCallState);
    if (!needsCapture) {
      if (logThis) {
        Logger::info(str::format("[SHOULD-CAPTURE-BUFFERS] verts=", vertCount, " REJECTED: shouldCaptureStatic=false"));
      }
      return false;
    }

    if (logThis) {
      Logger::info(str::format("[SHOULD-CAPTURE-BUFFERS] verts=", vertCount, " ACCEPTED!"));
    }

    // All checks passed - worth capturing buffers
    return true;
  }

bool D3D9Rtx::shouldCaptureFramebuffer() const {
    // Check if shader output capture is enabled (either all draws or specific hashes)
    static uint32_t s_logCount = 0;
    bool result = ShaderOutputCapturer::captureAllDraws() || !ShaderOutputCapturer::captureEnabledHashes().empty();

    if (++s_logCount <= 10) {
      Logger::info(str::format("[D3D9RTX-FRAMEBUFFER #", s_logCount, "] shouldCaptureFramebuffer() called - ",
                              "captureAllDraws=", ShaderOutputCapturer::captureAllDraws(),
                              " hashesSize=", ShaderOutputCapturer::captureEnabledHashes().size(),
                              " RETURNING: ", result));
    }

    return result;
  }

  Rc<DxvkImageView> D3D9Rtx::prepareFramebufferCapture(D3D9Surface* srcRenderTarget) {
    static uint32_t logCount = 0;
    ++logCount;

    if (!shouldCaptureFramebuffer() || srcRenderTarget == nullptr) {
      if (logCount <= 10) {
        Logger::info(str::format("[prepareFramebuffer-SKIP] shouldCaptureFramebuffer=", shouldCaptureFramebuffer(), " srcRenderTarget=", srcRenderTarget != nullptr ? "valid" : "null"));
      }
      return nullptr;
    }

    // CRITICAL PERF FIX: Check if this specific material needs capture
    // If the material is already cached (static material or not time for recapture),
    // return nullptr to skip framebuffer copy AND expensive GetImageView() call.
    // This prevents hundreds of GetImageView() calls per frame for cached materials.
    const bool needsCapture = ShaderOutputCapturer::shouldCaptureStatic(m_activeDrawCallState);
    if (logCount <= 10) {
      Logger::info(str::format("[prepareFramebuffer-CHECK] needsCapture=", needsCapture ? 1 : 0));
    }
    if (!needsCapture) {
      return nullptr;
    }

    // PERF: Only call expensive GetImageView() when we actually need to capture
    Rc<DxvkImageView> srcImageView = srcRenderTarget->GetImageView(false);

    // Get render target properties
    const auto& rtInfo = srcImageView->imageInfo();
    VkFormat format = rtInfo.format;
    VkExtent3D extent = rtInfo.extent;

    // NV-DXVK start: Validate and fix invalid render target properties
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
      Logger::warn(str::format(
        "D3D9Rtx: Fixing shader output capture texture with zero extent to 1x1:",
        "\n  Original Extent: ", extent.width, "x", extent.height, "x", extent.depth,
        "\n  Format: ", format));
      if (extent.width == 0) extent.width = 1;
      if (extent.height == 0) extent.height = 1;
      if (extent.depth == 0) extent.depth = 1;
    }

    if (format == VK_FORMAT_UNDEFINED) {
      Logger::warn(str::format(
        "D3D9Rtx: Fixing shader output capture texture with undefined format to R8G8B8A8_UNORM:",
        "\n  Extent: ", extent.width, "x", extent.height, "x", extent.depth));
      format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    // NV-DXVK end

    // Create a hash key for the cache based on format and dimensions
    XXH64_hash_t cacheKey = XXH64(&format, sizeof(format), 0);
    cacheKey = XXH64(&extent.width, sizeof(extent.width), cacheKey);
    cacheKey = XXH64(&extent.height, sizeof(extent.height), cacheKey);

    // Check if we have a cached texture for this configuration
    auto it = m_captureTextureCache.find(cacheKey);
    if (it != m_captureTextureCache.end()) {
      m_activeDrawCallState.capturedFramebufferOutput = it->second;
      return it->second;
    }

    // Create a new capture texture
    DxvkImageCreateInfo imageInfo;
    imageInfo.type = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.flags = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent = extent;
    imageInfo.numLayers = 1;
    imageInfo.mipLevels = 1;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    imageInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    Rc<DxvkImage> captureImage = m_parent->GetDXVKDevice()->createImage(
      imageInfo,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      DxvkMemoryStats::Category::RTXRenderTarget,
      "ShaderOutputCapture");

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;

    Rc<DxvkImageView> captureView = m_parent->GetDXVKDevice()->createImageView(captureImage, viewInfo);

    // Cache it
    m_captureTextureCache[cacheKey] = captureView;
    m_activeDrawCallState.capturedFramebufferOutput = captureView;

    // VRAM TRACKING: Log when we create new capture texture
    static uint32_t creationCount = 0;
    ++creationCount;
    const size_t estimatedVRAM = extent.width * extent.height * 4; // Assuming RGBA8
    const size_t totalCacheSize = m_captureTextureCache.size();

    Logger::warn(str::format("[VRAM-CAPTURE-TEXTURE] Created capture texture #", creationCount,
                              "\n  Resolution: ", extent.width, "x", extent.height,
                              "\n  Format: ", format,
                              "\n  Estimated VRAM: ", (estimatedVRAM / 1024 / 1024), " MB",
                              "\n  Total cached textures: ", totalCacheSize,
                              "\n  Cache key: 0x", std::hex, cacheKey, std::dec));

    return captureView;
  }

  bool D3D9Rtx::applyRenderTargetTextureReplacements() {
    // Check if we should do render target replacement for this draw call
    const int slot = m_activeDrawCallState.renderTargetReplacementSlot;
    if (slot < 0 || slot >= 16) {
      return false; // No replacement needed
    }

    // Get the current D3D9 state
    const auto& state = d3d9State();

    // Save the original texture at this slot and replace it with the captured framebuffer output
    m_replacedTextures[slot] = state.textures[slot];

    // Get the captured framebuffer output from the active draw call state
    if (m_activeDrawCallState.capturedFramebufferOutput != nullptr) {
      // We need to create a D3D9 texture wrapper for the captured framebuffer
      // For now, we'll just set it to nullptr to indicate replacement happened
      // The actual texture binding will be handled by the ShaderOutputCapturer
      m_parent->SetTexture(slot, nullptr);
      return true;
    }

    return false;
  }

  void D3D9Rtx::restoreReplacedTextures() {
    // Restore all replaced textures
    for (uint32_t slot = 0; slot < 16; ++slot) {
      if (m_replacedTextures[slot] != nullptr) {
        m_parent->SetTexture(slot, m_replacedTextures[slot]);
        m_replacedTextures[slot] = nullptr;
      }
    }
  }
}


