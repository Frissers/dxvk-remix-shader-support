#pragma once

#include "../../util/hlsl_decompiler_bridge.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

namespace dxvk {

/**
 * \brief Manages shader decompilation for RTX Remix
 *
 * This class handles async shader decompilation using background worker threads.
 * It tracks pending/completed counts for UI display.
 */
class ShaderCompatibilityManager {
public:
  /**
   * \brief Get the global instance for UI access
   */
  static ShaderCompatibilityManager* getInstance() { return s_instance; }

  /**
   * \brief Shader analysis information
   */
  struct ShaderCompatInfo {
    ShaderAnalysisResult analysisResult;
    std::atomic<bool> analyzed{false};

    // Custom copy constructor and assignment operator for atomics
    ShaderCompatInfo() = default;
    ShaderCompatInfo(const ShaderCompatInfo& other)
      : analysisResult(other.analysisResult)
      , analyzed(other.analyzed.load()) {}

    ShaderCompatInfo& operator=(const ShaderCompatInfo& other) {
      analysisResult = other.analysisResult;
      analyzed.store(other.analyzed.load());
      return *this;
    }
  };

  ShaderCompatibilityManager();
  ~ShaderCompatibilityManager();

  /**
   * \brief Check if all queued shaders have been analyzed
   */
  bool isAnalysisComplete() const { return m_pendingAnalysis.load() == 0; }

  /**
   * \brief Get number of shaders pending analysis
   */
  uint32_t getPendingAnalysisCount() const { return m_pendingAnalysis.load(); }

  /**
   * \brief Get total number of shaders analyzed
   */
  uint32_t getAnalyzedCount() const { return m_analyzedCount.load(); }

  /**
   * \brief Submit shader for async analysis (non-blocking)
   * Called on first use of a shader - analysis happens in background
   */
  void submitShaderForAsyncAnalysis(
    uint64_t shaderHash,
    const void* bytecode,
    size_t bytecodeLength,
    VkShaderStageFlagBits stage
  );

  /**
   * \brief Get analysis info for a shader (if available)
   */
  const ShaderCompatInfo* getShaderInfo(uint64_t shaderHash) const;

  /**
   * \brief Get the recommended albedo sampler index for a pixel shader
   * \param psHash Hash of the pixel shader
   * \returns The recommended sampler index (0-15) or -1 if unknown/not analyzed yet
   */
  int getRecommendedAlbedoSampler(uint64_t psHash) const;

  /**
   * \brief Register a texture as a detected render target
   * Called when decompiler indicates slot 0 isn't the albedo texture
   */
  void registerDetectedRenderTarget(uint64_t textureHash);

  /**
   * \brief Check if a texture has been detected as a render target
   */
  bool isDetectedRenderTarget(uint64_t textureHash) const;

  /**
   * \brief Get cached matrix registers from any analyzed shader with valid matrices
   * Scans all analyzed shaders and returns the first one with viewMatrixRegister >= 0
   * \param outViewReg Output: view matrix register (or -1 if none found)
   * \param outProjReg Output: projection matrix register (or -1 if none found)
   * \param outWorldReg Output: world matrix register (or -1 if none found)
   * \param outWvpReg Output: worldViewProj matrix register (or -1 if none found)
   * \returns true if a shader with valid matrix registers was found
   */
  bool findAnyShaderWithMatrixRegisters(int& outViewReg, int& outProjReg, int& outWorldReg, int& outWvpReg) const;

  /**
   * \brief Get default matrix registers from pre-scanned shaders
   */
  void getDefaultMatrixRegisters(int& outViewReg, int& outProjReg, int& outWorldReg, int& outWvpReg) const {
    outViewReg = m_defaultViewReg;
    outProjReg = m_defaultProjReg;
    outWorldReg = m_defaultWorldReg;
    outWvpReg = m_defaultWvpReg;
  }

  bool hasDefaultMatrixRegisters() const {
    return m_defaultViewReg >= 0 || m_defaultWorldReg >= 0 || m_defaultWvpReg >= 0;
  }

private:
  struct PendingShader {
    uint64_t hash;
    std::vector<uint8_t> bytecode;
    VkShaderStageFlagBits stage;
  };

  std::unordered_map<uint64_t, ShaderCompatInfo> m_shaderDatabase;
  mutable std::mutex m_databaseMutex;

  std::atomic<uint32_t> m_pendingAnalysis{0};
  std::atomic<uint32_t> m_analyzedCount{0};

  static inline ShaderCompatibilityManager* s_instance = nullptr;

  // Background worker pool for async shader analysis
  std::vector<std::thread> m_workerThreads;
  std::queue<PendingShader> m_asyncQueue;
  std::mutex m_asyncQueueMutex;
  std::condition_variable m_asyncQueueCV;
  std::atomic<bool> m_shutdownRequested{false};
  std::unordered_set<uint64_t> m_shadersSubmitted; // Track which shaders have been submitted
  std::mutex m_submittedMutex;

  // Auto-detected render targets (textures in slot 0 when albedo is elsewhere)
  std::unordered_set<uint64_t> m_detectedRenderTargets;
  mutable std::mutex m_detectedRTMutex;

  void processShaderAnalysis(uint64_t shaderHash, const std::vector<uint8_t>& bytecode, VkShaderStageFlagBits stage);

  // Background worker thread function
  void backgroundWorkerThread();

  // Start/stop background processing
  void startBackgroundProcessing();
  void stopBackgroundProcessing();

  // Pre-scan decompiled shaders on startup
  void prescanDecompiledShaders();

  // Default matrix registers from pre-scanned shaders
  int m_defaultViewReg = -1;
  int m_defaultProjReg = -1;
  int m_defaultWorldReg = -1;
  int m_defaultWvpReg = -1;
};

} // namespace dxvk
