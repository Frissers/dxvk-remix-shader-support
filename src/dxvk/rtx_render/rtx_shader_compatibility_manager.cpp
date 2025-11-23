#include "rtx_shader_compatibility_manager.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include "../../util/util_env.h"
#include <cstring>
#include <thread>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <regex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace dxvk {

ShaderCompatibilityManager::ShaderCompatibilityManager() {
  s_instance = this;

  // Pre-scan decompiled_shaders directory to get fallback matrix registers
  prescanDecompiledShaders();

  // Start background worker threads for async shader analysis
  startBackgroundProcessing();

  Logger::info("[ShaderCompat] Shader compatibility manager initialized with async processing");
}

void ShaderCompatibilityManager::prescanDecompiledShaders() {
  // Build path to decompiled_shaders directory
  std::filesystem::path exePath = env::getExePath();
  std::filesystem::path decompiledDir = exePath.parent_path() / "rtx-remix" / "decompiled_shaders";

  if (!std::filesystem::exists(decompiledDir)) {
    Logger::info(str::format("[ShaderCompat] No decompiled_shaders directory found at ", decompiledDir.string()));
    return;
  }

  Logger::info(str::format("[ShaderCompat] Pre-scanning decompiled shaders from ", decompiledDir.string()));

  int scannedCount = 0;
  int foundMatrixCount = 0;

  // Regex to parse matrix register info from shader header comments
  // Format: "// Matrix Registers: world=N view=N proj=N wvp=N"
  std::regex headerRegex(R"(Matrix Registers:\s*world=(-?\d+)\s+view=(-?\d+)\s+proj=(-?\d+)\s+wvp=(-?\d+))");

  // Regex to parse register declarations in HLSL
  // Format: "float4x4 vs_view : register(c4);"
  std::regex registerRegex(R"(float4x4\s+(\w+)\s*:\s*register\s*\(\s*c(\d+)\s*\))", std::regex::icase);

  for (const auto& entry : std::filesystem::directory_iterator(decompiledDir)) {
    if (!entry.is_regular_file()) continue;

    std::string filename = entry.path().filename().string();
    if (filename.substr(0, 3) != "vs_" || filename.substr(filename.length() - 4) != ".txt") continue;

    // Extract hash from filename: vs_0xHASH.txt
    std::string hashStr = filename.substr(3, filename.length() - 7); // Remove "vs_" and ".txt"
    uint64_t shaderHash = 0;
    try {
      shaderHash = std::stoull(hashStr, nullptr, 16);
    } catch (...) {
      continue; // Invalid hash format
    }

    // Read file content
    std::ifstream file(entry.path());
    if (!file.is_open()) continue;

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    scannedCount++;

    // Try to find matrix registers
    ShaderCompatInfo info;
    info.analyzed = true;
    info.analysisResult.success = true;

    // First try header comment format
    std::smatch headerMatch;
    if (std::regex_search(content, headerMatch, headerRegex)) {
      info.analysisResult.matrixInfo.worldMatrixRegister = std::stoi(headerMatch[1].str());
      info.analysisResult.matrixInfo.viewMatrixRegister = std::stoi(headerMatch[2].str());
      info.analysisResult.matrixInfo.projectionMatrixRegister = std::stoi(headerMatch[3].str());
      info.analysisResult.matrixInfo.worldViewProjMatrixRegister = std::stoi(headerMatch[4].str());
    }

    // Also parse HLSL register declarations for more specific names
    std::string::const_iterator searchStart = content.cbegin();
    std::smatch match;
    while (std::regex_search(searchStart, content.cend(), match, registerRegex)) {
      std::string varName = match[1].str();
      int regNum = std::stoi(match[2].str());

      // Convert to lowercase for matching
      std::transform(varName.begin(), varName.end(), varName.begin(), ::tolower);

      if (varName.find("viewproj") != std::string::npos || varName.find("worldviewproj") != std::string::npos) {
        info.analysisResult.matrixInfo.worldViewProjMatrixRegister = regNum;
      } else if (varName.find("worldview") != std::string::npos) {
        info.analysisResult.matrixInfo.worldViewMatrixRegister = regNum;
      } else if (varName.find("world") != std::string::npos && varName.find("cam") == std::string::npos) {
        info.analysisResult.matrixInfo.worldMatrixRegister = regNum;
      } else if (varName.find("view") != std::string::npos && varName.find("proj") == std::string::npos) {
        info.analysisResult.matrixInfo.viewMatrixRegister = regNum;
      } else if (varName.find("proj") != std::string::npos) {
        info.analysisResult.matrixInfo.projectionMatrixRegister = regNum;
      }

      searchStart = match.suffix().first;
    }

    // Only add if we found some matrix info
    bool hasMatrixInfo = info.analysisResult.matrixInfo.viewMatrixRegister >= 0 ||
                         info.analysisResult.matrixInfo.worldMatrixRegister >= 0 ||
                         info.analysisResult.matrixInfo.projectionMatrixRegister >= 0 ||
                         info.analysisResult.matrixInfo.worldViewProjMatrixRegister >= 0;

    if (hasMatrixInfo) {
      foundMatrixCount++;
      std::lock_guard<std::mutex> lock(m_databaseMutex);
      m_shaderDatabase[shaderHash] = info;

      // Update default fallback registers if we found better values
      if (info.analysisResult.matrixInfo.viewMatrixRegister >= 0 && m_defaultViewReg < 0) {
        m_defaultViewReg = info.analysisResult.matrixInfo.viewMatrixRegister;
      }
      if (info.analysisResult.matrixInfo.worldMatrixRegister >= 0 && m_defaultWorldReg < 0) {
        m_defaultWorldReg = info.analysisResult.matrixInfo.worldMatrixRegister;
      }
      if (info.analysisResult.matrixInfo.worldViewProjMatrixRegister >= 0 && m_defaultWvpReg < 0) {
        m_defaultWvpReg = info.analysisResult.matrixInfo.worldViewProjMatrixRegister;
      }
    }
  }

  Logger::info(str::format("[ShaderCompat] Pre-scan complete: ", scannedCount, " shaders scanned, ",
    foundMatrixCount, " with matrix info. Defaults: view=c", m_defaultViewReg,
    " world=c", m_defaultWorldReg, " wvp=c", m_defaultWvpReg));
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

    // Parse the cached file to extract albedoSampler and matrix register values
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
          }

          // Parse "Matrix Registers: world=N view=N proj=N wvp=N"
          pos = line.find("Matrix Registers:");
          if (pos != std::string::npos) {
            // Parse world=
            size_t worldPos = line.find("world=", pos);
            if (worldPos != std::string::npos) {
              worldPos += 6;
              info.analysisResult.matrixInfo.worldMatrixRegister = std::stoi(line.substr(worldPos));
            }
            // Parse view=
            size_t viewPos = line.find("view=", pos);
            if (viewPos != std::string::npos) {
              viewPos += 5;
              info.analysisResult.matrixInfo.viewMatrixRegister = std::stoi(line.substr(viewPos));
            }
            // Parse proj=
            size_t projPos = line.find("proj=", pos);
            if (projPos != std::string::npos) {
              projPos += 5;
              info.analysisResult.matrixInfo.projectionMatrixRegister = std::stoi(line.substr(projPos));
            }
            // Parse wvp=
            size_t wvpPos = line.find("wvp=", pos);
            if (wvpPos != std::string::npos) {
              wvpPos += 4;
              info.analysisResult.matrixInfo.worldViewProjMatrixRegister = std::stoi(line.substr(wvpPos));
            }

            Logger::info(str::format("[ShaderCompat] Loaded cached shader ", std::hex, shaderHash, std::dec,
              " matrixRegs: world=", info.analysisResult.matrixInfo.worldMatrixRegister,
              " view=", info.analysisResult.matrixInfo.viewMatrixRegister,
              " proj=", info.analysisResult.matrixInfo.projectionMatrixRegister,
              " wvp=", info.analysisResult.matrixInfo.worldViewProjMatrixRegister));
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

bool ShaderCompatibilityManager::findAnyShaderWithMatrixRegisters(int& outViewReg, int& outProjReg, int& outWorldReg, int& outWvpReg) const {
  // Cache the result to avoid repeated scanning
  static int s_cachedViewReg = -1;
  static int s_cachedProjReg = -1;
  static int s_cachedWorldReg = -1;
  static int s_cachedWvpReg = -1;
  static bool s_cacheValid = false;
  static bool s_loggedOnce = false;
  static uint32_t s_scanLogCount = 0;

  if (s_cacheValid) {
    outViewReg = s_cachedViewReg;
    outProjReg = s_cachedProjReg;
    outWorldReg = s_cachedWorldReg;
    outWvpReg = s_cachedWvpReg;
    return s_cachedViewReg >= 0;
  }

  std::lock_guard<std::mutex> lock(m_databaseMutex);

  // Log database stats periodically
  if (s_scanLogCount++ < 10) {
    uint32_t totalShaders = m_shaderDatabase.size();
    uint32_t analyzedShaders = 0;
    uint32_t successfulShaders = 0;
    uint32_t withViewMatrix = 0;
    uint32_t withProjMatrix = 0;
    for (const auto& [hash, info] : m_shaderDatabase) {
      if (info.analyzed) {
        analyzedShaders++;
        if (info.analysisResult.success) {
          successfulShaders++;
          if (info.analysisResult.matrixInfo.viewMatrixRegister >= 0) {
            withViewMatrix++;
          }
          if (info.analysisResult.matrixInfo.projectionMatrixRegister >= 0) {
            withProjMatrix++;
          }
        }
      }
    }
    Logger::info(str::format("[MATRIX-SCAN] Database: total=", totalShaders,
      " analyzed=", analyzedShaders, " successful=", successfulShaders,
      " withViewMatrix=", withViewMatrix, " withProjMatrix=", withProjMatrix));
  }

  // Scan for view and projection separately - they may be in different shaders
  int foundViewReg = -1;
  int foundProjReg = -1;
  int foundWorldReg = -1;
  int foundWvpReg = -1;

  for (const auto& [hash, info] : m_shaderDatabase) {
    if (info.analyzed && info.analysisResult.success) {
      const auto& matrixInfo = info.analysisResult.matrixInfo;

      // Track best view register found
      if (matrixInfo.viewMatrixRegister >= 0 && foundViewReg < 0) {
        foundViewReg = matrixInfo.viewMatrixRegister;
      }

      // Track best projection register found
      if (matrixInfo.projectionMatrixRegister >= 0 && foundProjReg < 0) {
        foundProjReg = matrixInfo.projectionMatrixRegister;
      }

      // Track best world register found
      if (matrixInfo.worldMatrixRegister >= 0 && foundWorldReg < 0) {
        foundWorldReg = matrixInfo.worldMatrixRegister;
      }

      // Track best worldViewProj register found
      if (matrixInfo.worldViewProjMatrixRegister >= 0 && foundWvpReg < 0) {
        foundWvpReg = matrixInfo.worldViewProjMatrixRegister;
      }

      // Early exit if we found everything
      if (foundViewReg >= 0 && foundProjReg >= 0 && foundWorldReg >= 0 && foundWvpReg >= 0) {
        break;
      }
    }
  }

  // If we found a view register, cache and return
  if (foundViewReg >= 0) {
    s_cachedViewReg = foundViewReg;
    s_cachedProjReg = foundProjReg;
    s_cachedWorldReg = foundWorldReg;
    s_cachedWvpReg = foundWvpReg;
    s_cacheValid = true;

    if (!s_loggedOnce) {
      Logger::info(str::format("[ShaderCompat] Found matrix registers from shader analysis:",
        " view=c", s_cachedViewReg,
        " proj=c", s_cachedProjReg,
        " world=c", s_cachedWorldReg,
        " wvp=c", s_cachedWvpReg));
      s_loggedOnce = true;
    }

    outViewReg = s_cachedViewReg;
    outProjReg = s_cachedProjReg;
    outWorldReg = s_cachedWorldReg;
    outWvpReg = s_cachedWvpReg;
    return true;
  }

  outViewReg = -1;
  outProjReg = -1;
  outWorldReg = -1;
  outWvpReg = -1;
  return false;
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
