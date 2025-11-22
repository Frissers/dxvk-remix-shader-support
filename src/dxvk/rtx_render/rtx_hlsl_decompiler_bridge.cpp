#include "rtx_hlsl_decompiler_bridge.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace dxvk {

// Simple cache for decompiled shaders
class ShaderDecompilationCache {
public:
  void put(uint64_t hash, const HlslDecompilerBridge::ShaderInfo& info) {
    m_cache[hash] = info;
  }

  std::optional<HlslDecompilerBridge::ShaderInfo> get(uint64_t hash) const {
    auto it = m_cache.find(hash);
    if (it != m_cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

private:
  std::unordered_map<uint64_t, HlslDecompilerBridge::ShaderInfo> m_cache;
};

HlslDecompilerBridge::HlslDecompilerBridge(const std::string& decompilerPath)
  : m_decompilerPath(decompilerPath)
  , m_cache(std::make_unique<ShaderDecompilationCache>()) {

  // If no path provided, use default location
  if (m_decompilerPath.empty()) {
    m_decompilerPath = "../HlslDecompiler/bin/Release/net8.0/HlslDecompiler.exe";
  }

  // Check if decompiler exists
  if (!std::filesystem::exists(m_decompilerPath)) {
    Logger::warn(str::format("[RTX Shader Compat] HlslDecompiler not found at: ", m_decompilerPath));
  }
}

HlslDecompilerBridge::~HlslDecompilerBridge() = default;

bool HlslDecompilerBridge::executeDecompiler(const std::string& inputPath, const std::string& outputPath) {
#ifdef _WIN32
  std::string cmdLine = m_decompilerPath + " \"" + inputPath + "\" \"" + outputPath + "\"";

  STARTUPINFOA si = {};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESHOWWINDOW;
  si.wShowWindow = SW_HIDE;

  PROCESS_INFORMATION pi = {};

  if (!CreateProcessA(nullptr, const_cast<char*>(cmdLine.c_str()), nullptr, nullptr, FALSE,
                      CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
    Logger::warn(str::format("[RTX Shader Compat] Failed to execute decompiler: ", GetLastError()));
    return false;
  }

  WaitForSingleObject(pi.hProcess, 10000); // 10 second timeout

  DWORD exitCode = 0;
  GetExitCodeProcess(pi.hProcess, &exitCode);

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return exitCode == 0;
#else
  // TODO: Implement for non-Windows platforms if needed
  Logger::warn("[RTX Shader Compat] HlslDecompiler only supported on Windows");
  return false;
#endif
}

std::optional<HlslDecompilerBridge::ShaderInfo> HlslDecompilerBridge::parseDecompiledHLSL(const std::string& hlslCode) {
  ShaderInfo info;
  info.decompiledHLSL = hlslCode;

  // Parse the HLSL code to extract information
  // Look for common matrix variables
  info.hasWorldMatrix = hlslCode.find("world") != std::string::npos ||
                        hlslCode.find("World") != std::string::npos ||
                        hlslCode.find("WORLD") != std::string::npos;

  info.hasViewMatrix = hlslCode.find("view") != std::string::npos ||
                       hlslCode.find("View") != std::string::npos ||
                       hlslCode.find("VIEW") != std::string::npos;

  info.hasProjectionMatrix = hlslCode.find("proj") != std::string::npos ||
                             hlslCode.find("Proj") != std::string::npos ||
                             hlslCode.find("PROJ") != std::string::npos ||
                             hlslCode.find("projection") != std::string::npos;

  info.hasWorldViewMatrix = hlslCode.find("worldview") != std::string::npos ||
                            hlslCode.find("WorldView") != std::string::npos ||
                            hlslCode.find("WORLDVIEW") != std::string::npos;

  info.hasWorldViewProjMatrix = hlslCode.find("worldviewproj") != std::string::npos ||
                                hlslCode.find("WorldViewProj") != std::string::npos ||
                                hlslCode.find("WORLDVIEWPROJ") != std::string::npos ||
                                hlslCode.find("wvp") != std::string::npos;

  // Check for transpose operations
  info.hasTransposeIssue = hlslCode.find("transpose") != std::string::npos;

  // Check for inverse operations
  info.hasInverseIssue = hlslCode.find("inverse") != std::string::npos;

  // Count texture samples
  size_t pos = 0;
  info.numTextureSamples = 0;
  while ((pos = hlslCode.find("tex2D", pos)) != std::string::npos) {
    info.numTextureSamples++;
    pos += 5;
  }

  // Check for texture coordinate transforms
  info.hasTexcoordTransform = hlslCode.find("TEXCOORD") != std::string::npos &&
                              (hlslCode.find("mul") != std::string::npos ||
                               hlslCode.find("*") != std::string::npos);

  // Check outputs
  info.outputsPosition = hlslCode.find("POSITION") != std::string::npos ||
                         hlslCode.find("SV_Position") != std::string::npos;
  info.outputsNormal = hlslCode.find("NORMAL") != std::string::npos;
  info.outputsTexcoord = hlslCode.find("TEXCOORD") != std::string::npos;

  return info;
}

std::optional<HlslDecompilerBridge::ShaderInfo> HlslDecompilerBridge::decompileShader(
  const void* bytecode,
  size_t bytecodeLength,
  uint64_t shaderHash
) {
  // Check cache first
  if (m_enableCaching) {
    if (auto cached = m_cache->get(shaderHash)) {
      if (m_verbose) {
        Logger::info(str::format("[RTX Shader Compat] Using cached shader info for hash: ", shaderHash));
      }
      return cached;
    }
  }

  // Check if decompiler exists
  if (!std::filesystem::exists(m_decompilerPath)) {
    return std::nullopt;
  }

  try {
    // Create temp directory for shader files
    std::filesystem::path tempDir = std::filesystem::temp_directory_path() / "remix_shaders";
    std::filesystem::create_directories(tempDir);

    // Write bytecode to temp file
    std::string inputFile = (tempDir / (std::to_string(shaderHash) + ".o")).string();
    std::string outputFile = (tempDir / (std::to_string(shaderHash) + ".hlsl")).string();

    {
      std::ofstream out(inputFile, std::ios::binary);
      out.write(static_cast<const char*>(bytecode), bytecodeLength);
    }

    // Execute decompiler
    if (!executeDecompiler(inputFile, outputFile)) {
      if (m_verbose) {
        Logger::warn(str::format("[RTX Shader Compat] Failed to decompile shader: ", shaderHash));
      }
      return std::nullopt;
    }

    // Read decompiled HLSL
    std::ifstream in(outputFile);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string hlslCode = buffer.str();

    // Parse HLSL
    auto info = parseDecompiledHLSL(hlslCode);

    // Cache result
    if (m_enableCaching && info) {
      m_cache->put(shaderHash, *info);
    }

    // Clean up temp files
    std::filesystem::remove(inputFile);
    std::filesystem::remove(outputFile);

    if (m_verbose && info) {
      Logger::info(str::format("[RTX Shader Compat] Successfully decompiled shader: ", shaderHash));
    }

    return info;

  } catch (const std::exception& e) {
    Logger::warn(str::format("[RTX Shader Compat] Exception during decompilation: ", e.what()));
    return std::nullopt;
  }
}

} // namespace dxvk
