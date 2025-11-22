#include "rtx_shader_compatibility_manager.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include "../../util/util_env.h"
#include <cstring>
#include <thread>
#include <algorithm>
#include <fstream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

namespace dxvk {

ShaderCompatibilityManager::ShaderCompatibilityManager() {
  s_instance = this;

  // Start background worker threads for async shader analysis
  startBackgroundProcessing();

  Logger::info("[ShaderCompat] Shader compatibility manager initialized with async processing");
}

ShaderCompatibilityManager::~ShaderCompatibilityManager() {
  // Stop background processing
  stopBackgroundProcessing();

  s_instance = nullptr;
  Logger::info("[ShaderCompat] Shader compatibility manager shut down");
}

void ShaderCompatibilityManager::submitShaderForAsyncAnalysis(
  uint64_t shaderHash,
  const void* bytecode,
  size_t bytecodeLength,
  VkShaderStageFlagBits stage
) {
  // Check if already analyzed
  {
    std::lock_guard<std::mutex> lock(m_databaseMutex);
    auto it = m_shaderDatabase.find(shaderHash);
    if (it != m_shaderDatabase.end() && it->second.analyzed) {
      return; // Already analyzed
    }
  }

  // Check if already submitted
  {
    std::lock_guard<std::mutex> lock(m_submittedMutex);
    if (!m_shadersSubmitted.insert(shaderHash).second) {
      return; // Already submitted for analysis
    }
  }

  Logger::info(str::format("[ShaderCompat] Submitting shader ", std::hex, shaderHash, std::dec,
    " for async analysis (", bytecodeLength, " bytes, stage=", (int)stage, ")"));

  // Create pending shader
  PendingShader pending;
  pending.hash = shaderHash;
  pending.bytecode.assign(
    static_cast<const uint8_t*>(bytecode),
    static_cast<const uint8_t*>(bytecode) + bytecodeLength
  );
  pending.stage = stage;

  // Add to async queue and increment pending counter
  {
    std::lock_guard<std::mutex> lock(m_asyncQueueMutex);
    m_asyncQueue.push(std::move(pending));
    m_pendingAnalysis++;  // Track pending analysis (will be decremented when complete)
  }

  // Notify worker threads
  m_asyncQueueCV.notify_one();
}

void ShaderCompatibilityManager::processShaderAnalysis(
  uint64_t shaderHash,
  const std::vector<uint8_t>& bytecode,
  VkShaderStageFlagBits stage
) {
  // Check disk cache first
  std::string shaderType = (stage == VK_SHADER_STAGE_VERTEX_BIT) ? "vs" : "ps";
  std::string cacheFilename = str::format(shaderType, "_0x", std::hex, shaderHash, ".txt");

  // Get game exe directory for cache (NOT .trex, the actual game directory)
  static std::string s_decompilerOutputDir;
  static bool s_dirInitialized = false;
  static std::mutex s_dirMutex;

  {
    std::lock_guard<std::mutex> lock(s_dirMutex);
    if (!s_dirInitialized) {
      s_dirInitialized = true;
      std::string basePath;
#ifdef _WIN32
      // Get game folder - exe might be in .trex subfolder, so go up if needed
      char exePath[MAX_PATH];
      GetModuleFileNameA(NULL, exePath, MAX_PATH);
      basePath = std::string(exePath);
      size_t lastSlash = basePath.find_last_of("\\/");
      if (lastSlash != std::string::npos)
        basePath = basePath.substr(0, lastSlash);
      // If we're in .trex folder, go up one level to actual game folder
      if (basePath.size() >= 5) {
        std::string lastDir = basePath.substr(basePath.size() - 5);
        if (lastDir == ".trex" || lastDir == "\\trex") {
          lastSlash = basePath.find_last_of("\\/");
          if (lastSlash != std::string::npos)
            basePath = basePath.substr(0, lastSlash);
        }
      }
#else
      basePath = ".";
#endif
      s_decompilerOutputDir = basePath + "\\rtx-remix\\decompiled_shaders";
      // Create directories
      std::filesystem::create_directories(s_decompilerOutputDir);
      Logger::info(str::format("[ShaderCompat] Decompiler output directory: ", s_decompilerOutputDir));
    }
  }

  std::string cachePath = s_decompilerOutputDir + "\\" + cacheFilename;

  // Check if already cached on disk
  if (std::filesystem::exists(cachePath)) {
    ShaderCompatInfo info;
    info.analyzed = true;
    info.analysisResult.success = true;
    info.analysisResult.albedoSamplerIndex = -1;  // Default to unknown

    // Parse the cached file to extract albedoSampler value
    try {
      std::ifstream inFile(cachePath);
      if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
          // Look for "albedoSampler=N" pattern
          size_t pos = line.find("albedoSampler=");
          if (pos != std::string::npos) {
            pos += 14;  // Length of "albedoSampler="
            int samplerIndex = std::stoi(line.substr(pos));
            info.analysisResult.albedoSamplerIndex = samplerIndex;
            Logger::info(str::format("[ShaderCompat] Loaded cached shader ", std::hex, shaderHash, std::dec,
                                    " albedoSampler=", samplerIndex));
            break;
          }
          // Stop parsing after first few lines (header only)
          if (line.find("// ===") != std::string::npos) {
            break;
          }
        }
        inFile.close();
      }
    } catch (const std::exception& e) {
      Logger::warn(str::format("[ShaderCompat] Failed to parse cached shader: ", e.what()));
    }

    if (info.analysisResult.albedoSamplerIndex < 0) {
      Logger::info(str::format("[ShaderCompat] Shader ", std::hex, shaderHash, std::dec, " cached but no albedo sampler found"));
    }

    // Update database
    {
      std::lock_guard<std::mutex> lock(m_databaseMutex);
      m_shaderDatabase[shaderHash] = info;
    }

    m_analyzedCount++;
    m_pendingAnalysis--;
    return;
  }

  Logger::info(str::format("[ShaderCompat] Analyzing shader ", std::hex, shaderHash, std::dec,
    " (", bytecode.size(), " bytes, stage=", (int)stage, ")"));

  ShaderCompatInfo info;

  // Analyze shader using the static decompiler bridge
  info.analysisResult = HlslDecompilerBridge::analyzeShader(bytecode.data(), bytecode.size());

  if (!info.analysisResult.success) {
    Logger::warn(str::format("[ShaderCompat] Failed to analyze shader ", std::hex, shaderHash, std::dec,
      ": ", info.analysisResult.errorMessage));
  } else {
    Logger::info(str::format("[ShaderCompat] Successfully analyzed shader ", std::hex, shaderHash, std::dec,
      " type=", info.analysisResult.shaderType,
      " model=", info.analysisResult.shaderModel));

    // Save to disk cache
    try {
      // Directory already created during initialization

      std::ofstream outFile(cachePath);
      if (outFile.is_open()) {
        outFile << "// " << (stage == VK_SHADER_STAGE_VERTEX_BIT ? "Vertex" : "Pixel")
                << " Shader 0x" << std::hex << shaderHash << std::dec << "\n";
        outFile << "// Type: " << info.analysisResult.shaderType
                << " Model: " << info.analysisResult.shaderModel << "\n";

        // Matrix info for vertex shaders
        if (stage == VK_SHADER_STAGE_VERTEX_BIT) {
          outFile << "// Matrix Registers: world=" << info.analysisResult.matrixInfo.worldMatrixRegister
                  << " view=" << info.analysisResult.matrixInfo.viewMatrixRegister
                  << " proj=" << info.analysisResult.matrixInfo.projectionMatrixRegister
                  << " wvp=" << info.analysisResult.matrixInfo.worldViewProjMatrixRegister << "\n";
        }

        // Sampler info
        outFile << "// Samplers: " << info.analysisResult.samplers.size()
                << " albedoSampler=" << info.analysisResult.albedoSamplerIndex << "\n";

        outFile << "\n// === DECOMPILED ASM ===\n";
        outFile << info.analysisResult.decompiledASM << "\n";
        outFile << "\n// === DECOMPILED HLSL ===\n";
        outFile << info.analysisResult.decompiledHLSL << "\n";

        outFile.close();
        Logger::info(str::format("[ShaderCompat] Saved decompiled shader to ", cachePath));
      }
    } catch (const std::exception& e) {
      Logger::warn(str::format("[ShaderCompat] Failed to save shader cache: ", e.what()));
    }
  }

  info.analyzed = true;

  // Update database
  {
    std::lock_guard<std::mutex> lock(m_databaseMutex);
    m_shaderDatabase[shaderHash] = info;
  }

  // Update counters
  m_analyzedCount++;
  m_pendingAnalysis--;

  Logger::info(str::format("[ShaderCompat] Completed shader ", std::hex, shaderHash, std::dec,
    " (analyzed=", m_analyzedCount.load(), " pending=", m_pendingAnalysis.load(), ")"));
}

const ShaderCompatibilityManager::ShaderCompatInfo* ShaderCompatibilityManager::getShaderInfo(uint64_t shaderHash) const {
  std::lock_guard<std::mutex> lock(m_databaseMutex);
  auto it = m_shaderDatabase.find(shaderHash);
  if (it != m_shaderDatabase.end() && it->second.analyzed) {
    return &it->second;
  }
  return nullptr;
}

int ShaderCompatibilityManager::getRecommendedAlbedoSampler(uint64_t psHash) const {
  std::lock_guard<std::mutex> lock(m_databaseMutex);
  auto it = m_shaderDatabase.find(psHash);
  if (it != m_shaderDatabase.end() && it->second.analyzed && it->second.analysisResult.success) {
    return it->second.analysisResult.albedoSamplerIndex;
  }
  return -1; // Not analyzed yet or analysis failed
}

void ShaderCompatibilityManager::registerDetectedRenderTarget(uint64_t textureHash) {
  std::lock_guard<std::mutex> lock(m_detectedRTMutex);
  if (m_detectedRenderTargets.insert(textureHash).second) {
    Logger::info(str::format("[ShaderCompat] Auto-detected render target: 0x", std::hex, textureHash, std::dec));
  }
}

bool ShaderCompatibilityManager::isDetectedRenderTarget(uint64_t textureHash) const {
  std::lock_guard<std::mutex> lock(m_detectedRTMutex);
  return m_detectedRenderTargets.count(textureHash) > 0;
}

void ShaderCompatibilityManager::startBackgroundProcessing() {
  // Create worker threads - use 16 workers to keep decompiler pipeline full
  const uint32_t workerCount = 16;

  Logger::info(str::format("[ShaderCompat] Starting ", workerCount, " background worker threads"));

  for (uint32_t i = 0; i < workerCount; ++i) {
    m_workerThreads.emplace_back([this]() {
      backgroundWorkerThread();
    });
  }
}

void ShaderCompatibilityManager::stopBackgroundProcessing() {
  // Signal shutdown
  m_shutdownRequested = true;

  // Wake up all worker threads
  m_asyncQueueCV.notify_all();

  // Wait for all workers to finish
  for (auto& thread : m_workerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  m_workerThreads.clear();
  Logger::info("[ShaderCompat] Background workers stopped");
}

void ShaderCompatibilityManager::backgroundWorkerThread() {
  Logger::info("[ShaderCompat] Background worker thread started");

  while (!m_shutdownRequested) {
    PendingShader shader;
    bool hasWork = false;

    // Wait for work
    {
      std::unique_lock<std::mutex> lock(m_asyncQueueMutex);

      // Wait for work or shutdown
      m_asyncQueueCV.wait(lock, [this]() {
        return !m_asyncQueue.empty() || m_shutdownRequested;
      });

      if (m_shutdownRequested && m_asyncQueue.empty()) {
        break;
      }

      if (!m_asyncQueue.empty()) {
        shader = std::move(m_asyncQueue.front());
        m_asyncQueue.pop();
        hasWork = true;
      }
    }

    // Process shader outside of lock
    if (hasWork) {
      processShaderAnalysis(shader.hash, shader.bytecode, shader.stage);
    }
  }

  Logger::info("[ShaderCompat] Background worker thread exiting");
}

} // namespace dxvk
