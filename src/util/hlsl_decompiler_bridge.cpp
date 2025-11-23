/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* RTX Remix HLSL Decompiler Bridge Implementation
*/

#include "hlsl_decompiler_bridge.h"
#include "log/log.h"
#include "util_env.h"
#include "sha1/sha1.h"

#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <array>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif

namespace dxvk {

  bool HlslDecompilerBridge::s_cachingEnabled = true;
  std::map<std::string, ShaderAnalysisResult> HlslDecompilerBridge::s_analysisCache;
  std::mutex HlslDecompilerBridge::s_cacheMutex;
  std::unordered_map<std::string, std::shared_ptr<std::mutex>> HlslDecompilerBridge::s_shaderMutexes;
  std::mutex HlslDecompilerBridge::s_mutexMapMutex;
  std::mutex HlslDecompilerBridge::s_semaphoreMutex;
  std::condition_variable HlslDecompilerBridge::s_semaphoreCV;
  int HlslDecompilerBridge::s_activeDecompilers = 0;
  std::atomic<uint32_t> HlslDecompilerBridge::s_pendingDecompilations{0};
  std::atomic<uint32_t> HlslDecompilerBridge::s_decompiledCount{0};

  std::string HlslDecompilerBridge::getDecompilerPath() {
    // First check environment variable
    std::string envPath = env::getEnvVar("REMIX_HLSL_DECOMPILER_PATH");
    if (!envPath.empty()) {
      Logger::info(str::format("[HlslDecompiler] Checking env path: ", envPath));
      if (std::filesystem::exists(envPath)) {
        Logger::info(str::format("[HlslDecompiler] Found decompiler at env path: ", envPath));
        return envPath;
      }
      Logger::warn(str::format("[HlslDecompiler] Env path does not exist: ", envPath));
    }

    // Default paths to check
    std::vector<std::string> defaultPaths = {
      "C:/Users/Friss/Documents/RTX_REMIX_RUNTIME/HlslDecompiler/bin/Release/net8.0/HlslDecompiler.exe",
      "C:/Users/Friss/Documents/RTX_REMIX_RUNTIME/HlslDecompiler/bin/Debug/net8.0/HlslDecompiler.exe",
      "./HlslDecompiler/bin/Release/net8.0/HlslDecompiler.exe",
      "./HlslDecompiler/bin/Debug/net8.0/HlslDecompiler.exe",
    };

    Logger::info("[HlslDecompiler] Searching for decompiler in default paths...");
    for (const auto& path : defaultPaths) {
      Logger::debug(str::format("[HlslDecompiler] Checking: ", path));
      if (std::filesystem::exists(path)) {
        Logger::info(str::format("[HlslDecompiler] Found decompiler at: ", path));
        return path;
      }
    }

    Logger::warn("[HlslDecompiler] Could not find HlslDecompiler.exe in any default path. Set REMIX_HLSL_DECOMPILER_PATH environment variable.");
    Logger::warn("[HlslDecompiler] Searched paths:");
    for (const auto& path : defaultPaths) {
      Logger::warn(str::format("[HlslDecompiler]   - ", path));
    }
    return "";
  }

  void HlslDecompilerBridge::setCachingEnabled(bool enabled) {
    s_cachingEnabled = enabled;
  }

  std::string HlslDecompilerBridge::computeBytecodeHash(const void* bytecode, size_t bytecodeLength) {
    SHA1_CTX ctx;
    uint8_t digest[SHA1_DIGEST_LENGTH];

    SHA1Init(&ctx);
    SHA1Update(&ctx, static_cast<const uint8_t*>(bytecode), bytecodeLength);
    SHA1Final(digest, &ctx);

    // Convert to hex string
    std::stringstream ss;
    for (int i = 0; i < SHA1_DIGEST_LENGTH; i++) {
      ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    return ss.str();
  }

  std::string HlslDecompilerBridge::writeBytecodeToTempFile(const void* bytecode, size_t bytecodeLength) {
    std::string tempDir = std::filesystem::temp_directory_path().string();
    std::string hash = computeBytecodeHash(bytecode, bytecodeLength);
    std::string tempFile = tempDir + "/remix_shader_" + hash + ".fxc";

    // Check if file already exists
    if (std::filesystem::exists(tempFile)) {
      return tempFile;
    }

    std::ofstream out(tempFile, std::ios::binary);
    if (!out) {
      Logger::err("[HlslDecompiler] Failed to create temporary file: " + tempFile);
      return "";
    }

    out.write(reinterpret_cast<const char*>(bytecode), bytecodeLength);
    out.close();

    return tempFile;
  }

  std::string HlslDecompilerBridge::readFileContents(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
      return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  bool HlslDecompilerBridge::executeDecompiler(const std::string& inputFile, const std::string& outputDir, bool doAstAnalysis) {
#ifdef _WIN32
    std::string decompilerPath = getDecompilerPath();
    if (decompilerPath.empty()) {
      return false;
    }

    // Build command line with full path to input file
    // Note: HlslDecompiler.exe expects: HlslDecompiler.exe <input.fxc>
    // It automatically creates .fx and .asm files in the same directory as the input
    std::string cmdLine = "\"" + decompilerPath + "\" \"" + inputFile + "\"";

    Logger::info("[HlslDecompiler] >>> STEP 1: About to execute decompiler");
    Logger::info(str::format("[HlslDecompiler] Command: ", cmdLine));
    Logger::info(str::format("[HlslDecompiler] Working dir: ", outputDir));

    // Create process
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;

    PROCESS_INFORMATION pi = {};

    Logger::info("[HlslDecompiler] >>> STEP 2: Calling CreateProcessA...");

    if (!CreateProcessA(
      nullptr,
      const_cast<char*>(cmdLine.c_str()),
      nullptr,
      nullptr,
      FALSE,
      CREATE_NO_WINDOW,
      nullptr,
      outputDir.c_str(),
      &si,
      &pi)) {
      DWORD error = GetLastError();
      Logger::err(str::format("[HlslDecompiler] CreateProcessA FAILED with error code: ", error));
      Logger::err(str::format("[HlslDecompiler] Command was: ", cmdLine));
      return false;
    }

    Logger::info(str::format("[HlslDecompiler] >>> STEP 3: CreateProcessA SUCCESS - PID: ", pi.dwProcessId));

    // Wait for process to complete (with timeout)
    // Use polling loop instead of single WaitForSingleObject to avoid Windows hang bug
    Logger::debug("[HlslDecompiler] >>> STEP 4: Starting wait loop with 10s timeout...");
    auto waitStartTime = std::chrono::steady_clock::now();

    DWORD result = WAIT_TIMEOUT;
    const int maxIterations = 500; // 500 iterations * 20ms = 10 seconds
    int iteration = 0;

    for (iteration = 0; iteration < maxIterations; iteration++) {
      result = WaitForSingleObject(pi.hProcess, 20); // 20ms timeout per iteration for faster response
      if (result != WAIT_TIMEOUT) {
        break; // Process finished or error
      }
      // Log progress every 50 iterations (every second)
      if ((iteration + 1) % 50 == 0) {
        Logger::debug(str::format("[HlslDecompiler] >>> Still waiting... (", (iteration + 1) / 50, "s elapsed)"));
      }
    }

    auto waitEndTime = std::chrono::steady_clock::now();
    auto waitDuration = std::chrono::duration_cast<std::chrono::milliseconds>(waitEndTime - waitStartTime).count();
    Logger::info(str::format("[HlslDecompiler] >>> STEP 5: Wait loop completed after ", waitDuration, "ms (", iteration + 1, " iterations)"));

    DWORD exitCode = 0;
    bool timedOut = false;

    if (result == WAIT_TIMEOUT) {
      Logger::warn("[HlslDecompiler] >>> STEP 6a: TIMEOUT - Terminating process...");
      TerminateProcess(pi.hProcess, 1);
      Logger::info("[HlslDecompiler] >>> TerminateProcess called, waiting for termination...");
      WaitForSingleObject(pi.hProcess, 1000); // Wait for termination
      Logger::info("[HlslDecompiler] >>> Process terminated");
      timedOut = true;
    } else if (result == WAIT_OBJECT_0) {
      Logger::info("[HlslDecompiler] >>> STEP 6b: Process completed normally");
    } else if (result == WAIT_FAILED) {
      DWORD waitError = GetLastError();
      Logger::err(str::format("[HlslDecompiler] >>> STEP 6c: WaitForSingleObject FAILED with error: ", waitError));
    } else {
      Logger::warn(str::format("[HlslDecompiler] >>> STEP 6d: Unexpected wait result: ", result));
    }

    Logger::info("[HlslDecompiler] >>> STEP 7: Getting exit code...");
    GetExitCodeProcess(pi.hProcess, &exitCode);
    Logger::info(str::format("[HlslDecompiler] >>> Exit code: ", exitCode));

    Logger::info("[HlslDecompiler] >>> STEP 8: Closing handles...");
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    Logger::info("[HlslDecompiler] >>> Handles closed");

    if (timedOut) {
      Logger::err(str::format("[HlslDecompiler] Decompiler timed out on shader: ", inputFile));
      return false;
    }

    Logger::info(str::format("[HlslDecompiler] >>> STEP 9: Process completed successfully with exit code: ", exitCode));

    if (exitCode != 0) {
      Logger::debug(str::format("[HlslDecompiler] Decompiler returned non-zero exit code: ", exitCode, " (may indicate partial decompilation)"));
      // Don't treat as fatal error - decompiler may still produce useful output files
      // The exit code 0xE0434352 indicates .NET exception, but files are still created
    }

    return true;
#else
    Logger::err("[HlslDecompiler] Not implemented for non-Windows platforms");
    return false;
#endif
  }

  ShaderMatrixInfo HlslDecompilerBridge::parseMatrixUsage(const std::string& hlsl, const std::string& asm_code) {
    ShaderMatrixInfo info;

    static uint32_t s_parseLogCount = 0;
    bool shouldLog = (s_parseLogCount++ < 5);

    if (shouldLog) {
      Logger::info(str::format("[PARSE-MATRIX] HLSL length=", hlsl.length(), " ASM length=", asm_code.length()));
    }

    // Common matrix naming patterns in shaders - ORDER MATTERS! More specific first
    std::vector<std::tuple<std::string, int*>> matrixPatterns = {
      {"worldviewproj", &info.worldViewProjMatrixRegister},
      {"worldviewprojection", &info.worldViewProjMatrixRegister},
      {"wvp", &info.worldViewProjMatrixRegister},
      {"worldview", &info.worldViewMatrixRegister},
      {"viewproj", &info.worldViewProjMatrixRegister},  // viewProj goes to wvp slot
      {"projection", &info.projectionMatrixRegister},
      {"proj", &info.projectionMatrixRegister},
      {"world", &info.worldMatrixRegister},
      {"view", &info.viewMatrixRegister},
    };

    // Parse constant declarations to find matrices
    // Pattern: float4x4 matrixName : register(c#);
    std::regex constantDecl(R"(float4x4\s+(\w+)\s*(?::\s*register\s*\(\s*c(\d+)\s*\))?)", std::regex::icase);
    std::smatch match;

    std::string::const_iterator searchStart(hlsl.cbegin());
    while (std::regex_search(searchStart, hlsl.cend(), match, constantDecl)) {
      std::string matrixName = match[1].str();
      std::string registerStr = match[2].str();

      // Convert to lowercase for comparison
      std::string matrixNameLower = matrixName;
      std::transform(matrixNameLower.begin(), matrixNameLower.end(), matrixNameLower.begin(), ::tolower);

      int registerNum = registerStr.empty() ? -1 : std::stoi(registerStr);

      if (shouldLog) {
        Logger::info(str::format("[PARSE-MATRIX] Found float4x4 '", matrixName, "' at register c", registerNum));
      }

      // Match against known patterns
      for (const auto& [pattern, targetReg] : matrixPatterns) {
        if (matrixNameLower.find(pattern) != std::string::npos) {
          if (*targetReg == -1) {
            *targetReg = registerNum;
            if (shouldLog) {
              Logger::info(str::format("[PARSE-MATRIX]   -> Matched pattern '", pattern, "', set to c", registerNum));
            }
          }
          break;
        }
      }

      searchStart = match.suffix().first;
    }

    // Detect viewProj from ASM: look for 4-register matrix multiply that outputs to o0 (position)
    // Pattern: mul/mad with c0-c3 followed by mov o0, result
    if (info.worldViewProjMatrixRegister == -1) {
      // Look for pattern: mad/mul operations followed by "mov o0"
      // The register block used right before "mov o0" is likely viewProj
      std::regex positionOutput(R"((?:mul|mad)\s+(\w+),.*c(\d+).*\n(?:(?:mul|mad)\s+\1,.*c(\d+).*\n)*.*mov\s+o0,\s*\1)", std::regex::icase);
      std::smatch posMatch;

      if (std::regex_search(asm_code, posMatch, positionOutput)) {
        int reg = std::stoi(posMatch[2].str());
        int alignedReg = reg / 4 * 4;  // Align to 4-register boundary
        info.worldViewProjMatrixRegister = alignedReg;
        if (shouldLog) {
          Logger::info(str::format("[PARSE-MATRIX] Detected viewProj at c", alignedReg, " from position output pattern"));
        }
      } else {
        // Fallback: look for c0-c3 usage pattern (common viewProj location)
        std::regex c0c3Pattern(R"((?:mul|mad)\s+\w+,.*c0.*\n.*(?:mul|mad)\s+\w+,.*c1.*\n.*(?:mul|mad)\s+\w+,.*c2.*\n.*(?:mul|mad)\s+\w+,.*c3)", std::regex::icase);
        if (std::regex_search(asm_code, c0c3Pattern)) {
          info.worldViewProjMatrixRegister = 0;
          if (shouldLog) {
            Logger::info("[PARSE-MATRIX] Detected viewProj at c0 from c0-c3 consecutive usage pattern");
          }
        }
      }
    }

    // If we couldn't find matrices by name, try to infer from ASM
    // Look for mul operations with c# registers that span 4 consecutive registers
    if (info.projectionMatrixRegister == -1 && info.viewMatrixRegister == -1) {
      std::regex mulConstant(R"(mul\s+\w+,\s*\w+,\s*c(\d+))", std::regex::icase);
      std::smatch asmMatch;

      std::string::const_iterator asmSearch(asm_code.cbegin());
      std::map<int, int> registerUsageCount;

      while (std::regex_search(asmSearch, asm_code.cend(), asmMatch, mulConstant)) {
        int reg = std::stoi(asmMatch[1].str());
        registerUsageCount[reg / 4 * 4]++; // Align to 4-register boundary
        asmSearch = asmMatch.suffix().first;
      }

      // Find the most frequently used 4-register block
      int maxCount = 0;
      int maxReg = -1;
      for (const auto& [reg, count] : registerUsageCount) {
        if (count > maxCount) {
          maxCount = count;
          maxReg = reg;
        }
      }

      if (maxReg != -1) {
        // First matrix block is likely projection or worldviewproj
        info.projectionMatrixRegister = maxReg;
      }
    }

    // Detect row-major vs column-major
    // Row-major: each register is a row (typical for D3D9)
    // Column-major: each register is a column
    // Heuristic: if we see mul(matrix, vector), it's row-major
    //            if we see mul(vector, matrix), it's column-major
    if (hlsl.find("mul(") != std::string::npos) {
      std::regex mulPattern(R"(mul\s*\(\s*(\w+)\s*,\s*(\w+)\s*\))");
      std::smatch mulMatch;
      std::string::const_iterator mulSearch(hlsl.cbegin());

      int rowMajorScore = 0;
      int colMajorScore = 0;

      while (std::regex_search(mulSearch, hlsl.cend(), mulMatch, mulPattern)) {
        std::string arg1 = mulMatch[1].str();
        std::string arg2 = mulMatch[2].str();

        // Check if arg1 or arg2 contains matrix names
        bool arg1IsMatrix = arg1.find("matrix") != std::string::npos ||
                            arg1.find("Mat") != std::string::npos ||
                            arg1.find("world") != std::string::npos ||
                            arg1.find("view") != std::string::npos ||
                            arg1.find("proj") != std::string::npos;
        bool arg2IsMatrix = arg2.find("matrix") != std::string::npos ||
                            arg2.find("Mat") != std::string::npos ||
                            arg2.find("world") != std::string::npos ||
                            arg2.find("view") != std::string::npos ||
                            arg2.find("proj") != std::string::npos;

        if (arg1IsMatrix && !arg2IsMatrix) {
          rowMajorScore++;
        } else if (!arg1IsMatrix && arg2IsMatrix) {
          colMajorScore++;
        }

        mulSearch = mulMatch.suffix().first;
      }

      info.isRowMajor = rowMajorScore > colMajorScore;
      info.isColumnMajor = colMajorScore > rowMajorScore;
    }

    return info;
  }

  std::vector<ShaderSamplerInfo> HlslDecompilerBridge::parseSamplerUsage(const std::string& hlsl) {
    std::vector<ShaderSamplerInfo> samplers;

    // Pattern: sampler2D name : register(sN);  OR  sampler2D name;
    std::regex samplerPattern(R"(sampler2D\s+(\w+)(?:\s*:\s*register\s*\(\s*s(\d+)\s*\))?)", std::regex::icase);
    std::smatch match;

    std::string::const_iterator searchStart(hlsl.cbegin());
    while (std::regex_search(searchStart, hlsl.cend(), match, samplerPattern)) {
      ShaderSamplerInfo info;
      info.name = match[1].str();

      if (match[2].matched) {
        info.registerIndex = std::stoi(match[2].str());
      }

      // Detect technical textures based on name patterns
      std::string nameLower = info.name;
      std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);

      info.isTechnicalTexture =
        nameLower.find("depth") != std::string::npos ||
        nameLower.find("noise") != std::string::npos ||
        nameLower.find("blur") != std::string::npos ||
        nameLower.find("mipcolor") != std::string::npos ||
        nameLower.find("fullcolor") != std::string::npos ||
        nameLower.find("shadow") != std::string::npos ||
        nameLower.find("normal") != std::string::npos ||  // Normals buffer (not normal map)
        nameLower.find("scenedepth") != std::string::npos ||
        nameLower.find("minmax") != std::string::npos;

      // Detect probable albedo textures
      info.isProbablyAlbedo =
        nameLower.find("albedo") != std::string::npos ||
        nameLower.find("diffuse") != std::string::npos ||
        nameLower.find("color") != std::string::npos ||
        nameLower.find("base") != std::string::npos ||
        info.name == "texture0";  // texture0 is often the main texture

      samplers.push_back(info);
      searchStart = match.suffix().first;
    }

    return samplers;
  }

  ShaderTexCoordInfo HlslDecompilerBridge::parseTexCoordUsage(const std::string& hlsl, const std::string& asm_code) {
    ShaderTexCoordInfo info;

    // DEBUG: Log what we're analyzing
    Logger::debug("[HlslDecompiler] parseTexCoordUsage - starting analysis");
    Logger::debug(str::format("[HlslDecompiler] HLSL length: ", hlsl.length(), " ASM length: ", asm_code.length()));

    // Parse texture coordinate usage from HLSL
    // Look for TEXCOORD# semantics in input and usage
    std::regex texcoordPattern(R"(TEXCOORD(\d+))", std::regex::icase);

    // Enhanced sampler pattern - match both explicit register and without
    std::regex samplerPattern(R"(sampler(?:2D)?\s+(\w+)(?:\s*:\s*register\s*\(\s*s(\d+)\s*\))?)", std::regex::icase);
    std::regex tex2dPattern(R"(tex2D\s*\(\s*(\w+)\s*,\s*([^)]+)\))", std::regex::icase);

    // Find all samplers
    std::map<std::string, int> samplerMap;
    std::smatch match;
    std::string::const_iterator searchStart(hlsl.cbegin());

    int samplerCount = 0;
    while (std::regex_search(searchStart, hlsl.cend(), match, samplerPattern)) {
      std::string samplerName = match[1].str();
      int registerIndex = match[2].matched ? std::stoi(match[2].str()) : samplerCount;
      samplerMap[samplerName] = registerIndex;

      Logger::debug(str::format("[HlslDecompiler] Found sampler: ", samplerName, " -> s", registerIndex));
      samplerCount++;
      searchStart = match.suffix().first;
    }

    // Find tex2D calls and match them to texcoords
    searchStart = hlsl.cbegin();
    std::map<std::string, ShaderTexCoordInfo::TexCoordUsage> texCoordMap;

    Logger::debug("[HlslDecompiler] Searching for tex2D calls...");
    int tex2dCallCount = 0;

    while (std::regex_search(searchStart, hlsl.cend(), match, tex2dPattern)) {
      std::string samplerName = match[1].str();
      std::string texcoordExpression = match[2].str();

      tex2dCallCount++;
      Logger::debug(str::format("[HlslDecompiler] tex2D call #", tex2dCallCount, ": sampler=", samplerName, " texcoord=\"", texcoordExpression, "\""));

      // Trim whitespace
      texcoordExpression.erase(0, texcoordExpression.find_first_not_of(" \t\n\r"));
      texcoordExpression.erase(texcoordExpression.find_last_not_of(" \t\n\r") + 1);

      // Try to extract base texcoord variable name (handle member access like "i.texcoord2.xy")
      std::regex texcoordVarExtract(R"((\w+\.\w+)\.)"  // Member access pattern: struct.member.swizzle
                                     "|"              // OR
                                     R"((\w+\.))");    // Simple member access: struct.swizzle
      std::smatch varMatch;
      std::string texcoordVar;
      std::string fullVar;  // Full variable before swizzle

      if (std::regex_search(texcoordExpression, varMatch, texcoordVarExtract)) {
        if (varMatch[1].matched) {
          fullVar = varMatch[1].str();  // e.g., "i.texcoord2" from "i.texcoord2.xy"
        } else if (varMatch[2].matched) {
          fullVar = varMatch[2].str();
          fullVar.pop_back();  // Remove trailing '.'
        }

        // Extract just the member name (after the dot) for semantic lookup
        size_t dotPos = fullVar.find('.');
        if (dotPos != std::string::npos && dotPos + 1 < fullVar.length()) {
          texcoordVar = fullVar.substr(dotPos + 1);  // e.g., "texcoord2" from "i.texcoord2"
        }

        Logger::debug(str::format("[HlslDecompiler] Extracted texcoord variable: fullVar='", fullVar, "', member='", texcoordVar, "'"));
      }

      // Fallback: if no member access, just get first word
      if (texcoordVar.empty()) {
        std::regex simpleExtract(R"((\w+))");
        if (std::regex_search(texcoordExpression, varMatch, simpleExtract)) {
          texcoordVar = varMatch[1].str();
          Logger::debug(str::format("[HlslDecompiler] Extracted simple texcoord variable: ", texcoordVar));
        }
      }

      if (!texcoordVar.empty()) {
        // METHOD 1: Try to find TEXCOORD semantic in variable declaration
        // Pattern: varName : TEXCOORD# or varName : TEXCOORD (without number = TEXCOORD0)
        std::regex texcoordVarPattern(texcoordVar + R"(\s*:\s*TEXCOORD(\d*))", std::regex::icase);
        std::smatch texcoordMatch;

        int texcoordIndex = -1;

        if (std::regex_search(hlsl, texcoordMatch, texcoordVarPattern)) {
          // If no number captured, default to TEXCOORD0
          if (texcoordMatch[1].str().empty()) {
            texcoordIndex = 0;
            Logger::debug("[HlslDecompiler] Found TEXCOORD semantic without number: defaulting to TEXCOORD0");
          } else {
            texcoordIndex = std::stoi(texcoordMatch[1].str());
            Logger::debug(str::format("[HlslDecompiler] Found TEXCOORD semantic: TEXCOORD", texcoordIndex));
          }
        }

        // METHOD 2: Check if variable name contains texcoord index (e.g., "v0", "texcoord0", "tc1")
        // Also handle "texcoord" without number = TEXCOORD0
        if (texcoordIndex == -1) {
          std::regex varIndexPattern(R"((?:v|texcoord|tc)(\d+))", std::regex::icase);
          std::smatch varIndexMatch;
          if (std::regex_search(texcoordVar, varIndexMatch, varIndexPattern)) {
            texcoordIndex = std::stoi(varIndexMatch[1].str());
            Logger::debug(str::format("[HlslDecompiler] Inferred TEXCOORD index from variable name: ", texcoordIndex));
          } else {
            // Check if variable is just "texcoord" or "tc" without a number -> default to 0
            std::string varLower = texcoordVar;
            std::transform(varLower.begin(), varLower.end(), varLower.begin(), ::tolower);
            if (varLower == "texcoord" || varLower == "tc" || varLower == "uv") {
              texcoordIndex = 0;
              Logger::debug("[HlslDecompiler] Variable name suggests TEXCOORD0 (no explicit number)");
            }
          }
        }

        // METHOD 3: Check ASM for dcl_texcoord v#
        if (texcoordIndex == -1 && !asm_code.empty()) {
          // Look for dcl_texcoord v# in ASM that corresponds to this variable
          std::regex asmDclPattern(R"(dcl_texcoord(\d*)?\s+v(\d+))", std::regex::icase);
          std::smatch asmDclMatch;
          std::string::const_iterator asmDclSearch(asm_code.cbegin());
          int foundIndex = 0;
          while (std::regex_search(asmDclSearch, asm_code.cend(), asmDclMatch, asmDclPattern)) {
            // Use the register number as texcoord index
            int regNum = std::stoi(asmDclMatch[2].str());
            // If there's an explicit TEXCOORD number, use that; otherwise use register number
            if (asmDclMatch[1].matched) {
              texcoordIndex = std::stoi(asmDclMatch[1].str());
            } else {
              texcoordIndex = regNum;
            }
            Logger::debug(str::format("[HlslDecompiler] Found dcl_texcoord in ASM: v", regNum, " -> TEXCOORD", texcoordIndex));
            foundIndex++;
            asmDclSearch = asmDclMatch.suffix().first;
          }
        }

        if (texcoordIndex >= 0) {
          Logger::debug(str::format("[HlslDecompiler] Processing TEXCOORD", texcoordIndex));

          ShaderTexCoordInfo::TexCoordUsage usage;
          usage.texCoordIndex = texcoordIndex;

          if (samplerMap.find(samplerName) != samplerMap.end()) {
            usage.samplerIndex = samplerMap[samplerName];
          }

          // **ENHANCED UV TRANSFORMATION DETECTION**
          // Detect various transformation patterns:

          // 1. Matrix multiplication: mul(matrix, uv) or mul(uv, matrix)
          std::regex mulPattern(R"(mul\s*\([^,]+,\s*[^)]+\))");
          if (std::regex_search(texcoordExpression, mulPattern)) {
            usage.isTransformed = true;
            usage.operations.push_back("matrix_multiplication");

            // Try to find which constant register holds the matrix
            std::regex constPattern(R"(c(\d+))");
            std::smatch constMatch;
            if (std::regex_search(texcoordExpression, constMatch, constPattern)) {
              usage.transformMatrixRegister = std::stoi(constMatch[1].str());
            }
          }

          // 2. Addition/subtraction with constants (scrolling/offset UVs)
          if (texcoordExpression.find("+") != std::string::npos ||
              texcoordExpression.find("-") != std::string::npos) {
            usage.operations.push_back("offset");

            // Check if offset involves camera position or time
            if (texcoordExpression.find("cameraPos") != std::string::npos ||
                texcoordExpression.find("eyePos") != std::string::npos ||
                texcoordExpression.find("viewPos") != std::string::npos) {
              usage.operations.push_back("camera_position_offset");
              info.hasViewSpaceTexCoords = true;
            }
          }

          // 3. Multiplication with scalars (tiling)
          if (texcoordExpression.find("*") != std::string::npos &&
              !std::regex_search(texcoordExpression, mulPattern)) {
            usage.operations.push_back("scale");
          }

          // 4. Swizzling operations (.xy, .zw, .xz, etc)
          if (texcoordExpression.find(".") != std::string::npos) {
            usage.operations.push_back("swizzle");
          }

          // 5. Division operations (projected textures)
          if (texcoordExpression.find("/") != std::string::npos) {
            usage.isProjected = true;
            usage.operations.push_back("projection");
          }

          // 6. Function calls (custom UV manipulation)
          std::regex funcCallPattern(R"(\w+\s*\()");
          if (std::regex_search(texcoordExpression, funcCallPattern)) {
            usage.operations.push_back("function_transform");
          }

          // **DETECT WORLD-SPACE UV GENERATION**
          // Check if UV is computed from world position instead of input texcoords
          std::string exprLower = texcoordExpression;
          std::transform(exprLower.begin(), exprLower.end(), exprLower.begin(), ::tolower);

          if (exprLower.find("worldpos") != std::string::npos ||
              exprLower.find("world.pos") != std::string::npos ||
              exprLower.find("position.") != std::string::npos ||
              exprLower.find("pos.") != std::string::npos) {
            usage.operations.push_back("world_space_generation");
            info.hasWorldSpaceTexCoords = true;
          }

          texCoordMap[texcoordVar] = usage;
        }
      }

      searchStart = match.suffix().first;
    }

    // Convert map to vector
    for (const auto& [name, usage] : texCoordMap) {
      info.texCoords.push_back(usage);
    }

    // **DETECT DOT-PRODUCT UV TRANSFORMS IN VERTEX SHADERS**
    // Pattern: o.texcoord# = dot(something, vs_RegisterName)
    // This is how LEGO Batman 2 (and many games) apply UV scrolling/transforms
    std::regex dotProductHlslPattern(R"(o\.texcoord(\d*)\.([xy])\s*=\s*dot\s*\([^,]+,\s*(\w+))", std::regex::icase);
    searchStart = hlsl.cbegin();

    Logger::debug("[HlslDecompiler] Searching for dot-product UV transforms in HLSL...");

    while (std::regex_search(searchStart, hlsl.cend(), match, dotProductHlslPattern)) {
      std::string texcoordIndexStr = match[1].str();
      std::string component = match[2].str();  // x or y
      std::string matrixVarName = match[3].str();

      int texcoordIndex = texcoordIndexStr.empty() ? 0 : std::stoi(texcoordIndexStr);

      Logger::info(str::format("[HlslDecompiler] Found dot-product UV transform: o.texcoord", texcoordIndex,
        ".", component, " = dot(..., ", matrixVarName, ")"));

      // Check if we already have this texcoord index
      bool found = false;
      for (auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          tc.isTransformed = true;
          if (std::find(tc.operations.begin(), tc.operations.end(), "dot_product_transform") == tc.operations.end()) {
            tc.operations.push_back("dot_product_transform");
          }

          // Try to find the register number for this matrix variable
          // Pattern: float4 matrixName : register(c#);
          std::regex registerPattern(matrixVarName + R"(\s*:\s*register\s*\(\s*c(\d+)\s*\))", std::regex::icase);
          std::smatch regMatch;
          if (std::regex_search(hlsl, regMatch, registerPattern)) {
            tc.transformMatrixRegister = std::stoi(regMatch[1].str());
            Logger::info(str::format("[HlslDecompiler] Matrix ", matrixVarName, " is at register c", tc.transformMatrixRegister));
          }
          break;
        }
      }

      if (!found) {
        // Add new texcoord entry for vertex shader output
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        usage.isTransformed = true;
        usage.operations.push_back("dot_product_transform");

        // Try to find the register number
        std::regex registerPattern(matrixVarName + R"(\s*:\s*register\s*\(\s*c(\d+)\s*\))", std::regex::icase);
        std::smatch regMatch;
        if (std::regex_search(hlsl, regMatch, registerPattern)) {
          usage.transformMatrixRegister = std::stoi(regMatch[1].str());
          Logger::info(str::format("[HlslDecompiler] Matrix ", matrixVarName, " is at register c", usage.transformMatrixRegister));
        }

        info.texCoords.push_back(usage);
        Logger::info(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " from dot-product transform"));
      }

      searchStart = match.suffix().first;
    }

    // **DETECT OUTPUT TEXCOORDS IN VERTEX SHADERS**
    // Pattern: VS_OUT { ... float# texcoord : TEXCOORD#; ... }
    // Vertex shaders output texcoords that pixel shaders consume
    std::regex outputTexcoordPattern(R"(\s+(?:float\d*|half\d*)\s+texcoord(\d*)\s*:\s*TEXCOORD(\d*))", std::regex::icase);
    searchStart = hlsl.cbegin();

    Logger::debug("[HlslDecompiler] Searching for output TEXCOORD declarations in vertex shader...");

    while (std::regex_search(searchStart, hlsl.cend(), match, outputTexcoordPattern)) {
      std::string varIndexStr = match[1].str();
      std::string semanticIndexStr = match[2].str();

      // Use semantic index if present, otherwise use variable index, otherwise 0
      int texcoordIndex;
      if (!semanticIndexStr.empty()) {
        texcoordIndex = std::stoi(semanticIndexStr);
      } else if (!varIndexStr.empty()) {
        texcoordIndex = std::stoi(varIndexStr);
      } else {
        texcoordIndex = 0;
      }

      Logger::debug(str::format("[HlslDecompiler] Found output TEXCOORD", texcoordIndex, " declaration in VS_OUT"));

      // Check if we already have this texcoord
      bool found = false;
      for (const auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          break;
        }
      }

      if (!found) {
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        info.texCoords.push_back(usage);
        Logger::debug(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " from vertex shader output"));
      }

      searchStart = match.suffix().first;
    }

    // **ADDITIONAL DETECTION FROM ASM CODE**
    // Sometimes HLSL decompilation misses things, so check ASM too
    // Look for texture coordinate operations in assembly
    Logger::debug("[HlslDecompiler] Checking ASM for additional texcoord usage...");

    // Pattern 1: dcl_texcoord v# (pixel shader input) or dcl_texcoord o# (vertex shader output)
    // Handle both "dcl_texcoord v#" and "dcl_texcoord0 v#" formats
    std::regex asmDclTexPattern(R"(dcl_texcoord(\d*)\s+([vo])(\d+))", std::regex::icase);
    std::smatch asmMatch;
    std::string::const_iterator asmSearch(asm_code.cbegin());

    while (std::regex_search(asmSearch, asm_code.cend(), asmMatch, asmDclTexPattern)) {
      std::string registerType = asmMatch[2].str();  // "v" or "o"
      int regNum = std::stoi(asmMatch[3].str());

      // If TEXCOORD number is specified, use that; otherwise use register number
      int texcoordIndex;
      if (asmMatch[1].str().empty()) {
        texcoordIndex = regNum;  // No explicit number means use register number
        Logger::debug(str::format("[HlslDecompiler] Found dcl_texcoord ", registerType, regNum, " (no index) -> TEXCOORD", texcoordIndex));
      } else {
        texcoordIndex = std::stoi(asmMatch[1].str());
        Logger::debug(str::format("[HlslDecompiler] Found dcl_texcoord", texcoordIndex, " ", registerType, regNum));
      }

      // Check if this texcoord is already in our list
      bool found = false;
      for (const auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          break;
        }
      }

      if (!found) {
        // Add missing texcoord detected from ASM
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        info.texCoords.push_back(usage);
        Logger::debug(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " from ASM dcl_texcoord"));
      }

      asmSearch = asmMatch.suffix().first;
    }

    // Pattern 2: texcoord instruction (e.g., texcoord r0, v1)
    std::regex asmTexInstrPattern(R"(texcoord\s+r\d+(?:\.\w+)?,\s*v(\d+))", std::regex::icase);
    asmSearch = asm_code.cbegin();

    while (std::regex_search(asmSearch, asm_code.cend(), asmMatch, asmTexInstrPattern)) {
      int texcoordIndex = std::stoi(asmMatch[1].str());

      // Check if this texcoord is already in our list
      bool found = false;
      for (const auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          break;
        }
      }

      if (!found) {
        // Add missing texcoord detected from ASM
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        info.texCoords.push_back(usage);
        Logger::debug(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " from ASM texcoord instruction"));
      }

      asmSearch = asmMatch.suffix().first;
    }

    // Pattern 3: tex2D instruction in ASM (e.g., texld r0, v1, s0)
    std::regex asmTexldPattern(R"(texld\s+r\d+,\s*v(\d+),\s*s(\d+))", std::regex::icase);
    asmSearch = asm_code.cbegin();

    while (std::regex_search(asmSearch, asm_code.cend(), asmMatch, asmTexldPattern)) {
      int texcoordIndex = std::stoi(asmMatch[1].str());
      int samplerIndex = std::stoi(asmMatch[2].str());

      // Check if this texcoord is already in our list
      bool found = false;
      for (auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          if (tc.samplerIndex == -1) {
            tc.samplerIndex = samplerIndex;
          }
          break;
        }
      }

      if (!found) {
        // Add missing texcoord detected from ASM
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        usage.samplerIndex = samplerIndex;
        info.texCoords.push_back(usage);
        Logger::debug(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " -> s", samplerIndex, " from ASM texld"));
      }

      asmSearch = asmMatch.suffix().first;
    }

    // Pattern 4: dp3 (dot product) instruction with output texcoord (e.g., dp3 o3.x, r0.xyz, c23.xyw)
    // This is THE CRITICAL PATTERN for UV scrolling in LEGO Batman 2 and similar games
    std::regex asmDp3OutputPattern(R"(dp3\s+o(\d+)\.([xy]),\s*[^,]+,\s*c(\d+))", std::regex::icase);
    asmSearch = asm_code.cbegin();

    Logger::debug("[HlslDecompiler] Searching for dp3 UV transforms in ASM...");

    while (std::regex_search(asmSearch, asm_code.cend(), asmMatch, asmDp3OutputPattern)) {
      int outputRegister = std::stoi(asmMatch[1].str());
      std::string component = asmMatch[2].str();  // x or y
      int constantRegister = std::stoi(asmMatch[3].str());

      Logger::info(str::format("[HlslDecompiler] Found dp3 UV transform in ASM: o", outputRegister,
        ".", component, " = dp3(..., c", constantRegister, ")"));

      // Output register o# corresponds to TEXCOORD# in many cases
      int texcoordIndex = outputRegister;

      // Check if we already have this texcoord
      bool found = false;
      for (auto& tc : info.texCoords) {
        if (tc.texCoordIndex == texcoordIndex) {
          found = true;
          tc.isTransformed = true;
          if (std::find(tc.operations.begin(), tc.operations.end(), "dot_product_transform") == tc.operations.end()) {
            tc.operations.push_back("dot_product_transform");
          }
          // Store the constant register that affects this UV
          // IMPORTANT: For UVs transformed by dp3, BOTH X and Y might use DIFFERENT registers (c23 for X, c24 for Y)
          // We store the FIRST register in transformMatrixRegister for backward compatibility,
          // but the UV profile builder will detect both registers by scanning the ASM separately
          if (tc.transformMatrixRegister == -1) {
            tc.transformMatrixRegister = constantRegister;
          }
          break;
        }
      }

      if (!found) {
        // Add new texcoord entry
        ShaderTexCoordInfo::TexCoordUsage usage;
        usage.texCoordIndex = texcoordIndex;
        usage.isTransformed = true;
        usage.operations.push_back("dot_product_transform");
        usage.transformMatrixRegister = constantRegister;
        info.texCoords.push_back(usage);
        Logger::info(str::format("[HlslDecompiler] Added TEXCOORD", texcoordIndex, " from ASM dp3 with transform register c", constantRegister));
      }

      asmSearch = asmMatch.suffix().first;
    }

    // Check for world/view space texcoords
    info.hasWorldSpaceTexCoords = info.hasWorldSpaceTexCoords ||
                                    hlsl.find("worldPos") != std::string::npos ||
                                    hlsl.find("WorldPos") != std::string::npos;
    info.hasViewSpaceTexCoords = info.hasViewSpaceTexCoords ||
                                   hlsl.find("viewPos") != std::string::npos ||
                                   hlsl.find("ViewPos") != std::string::npos;

    // FINAL DEBUG SUMMARY
    Logger::info(str::format("[HlslDecompiler] parseTexCoordUsage complete: Found ", info.texCoords.size(), " texture coordinates"));
    for (size_t i = 0; i < info.texCoords.size(); i++) {
      const auto& tc = info.texCoords[i];
      Logger::debug(str::format("[HlslDecompiler]   TEXCOORD", tc.texCoordIndex,
        " -> sampler ", tc.samplerIndex,
        " | transformed=", tc.isTransformed ? "YES" : "NO",
        " | ops=", tc.operations.size()));
    }

    if (info.texCoords.empty()) {
      Logger::warn("[HlslDecompiler] WARNING: No texture coordinates detected! This may cause UV neutralization to fail.");
      Logger::warn("[HlslDecompiler] HLSL snippet:");
      Logger::warn(hlsl.substr(0, std::min(size_t(500), hlsl.length())));
      Logger::warn("[HlslDecompiler] ASM snippet:");
      Logger::warn(asm_code.substr(0, std::min(size_t(500), asm_code.length())));
    }

    return info;
  }

  ShaderAnalysisResult HlslDecompilerBridge::analyzeShader(const void* bytecode, size_t bytecodeLength) {
    ShaderAnalysisResult result;

    std::string hash = computeBytecodeHash(bytecode, bytecodeLength);

    // Quick cache check without locking
    if (s_cachingEnabled) {
      std::lock_guard<std::mutex> cacheLock(s_cacheMutex);
      auto it = s_analysisCache.find(hash);
      if (it != s_analysisCache.end()) {
        Logger::debug("[HlslDecompiler] Using cached analysis result");
        return it->second;
      }
    }

    // Get or create a mutex for this specific shader
    // This allows DIFFERENT shaders to be analyzed in parallel
    // while ensuring the SAME shader is only analyzed once
    std::shared_ptr<std::mutex> shaderMutex;
    {
      std::lock_guard<std::mutex> mapLock(s_mutexMapMutex);
      auto it = s_shaderMutexes.find(hash);
      if (it == s_shaderMutexes.end()) {
        // First thread to request this shader creates its mutex
        shaderMutex = std::make_shared<std::mutex>();
        s_shaderMutexes[hash] = shaderMutex;
      } else {
        shaderMutex = it->second;
      }
    }

    // Lock THIS shader's mutex (other shaders can still proceed in parallel)
    std::lock_guard<std::mutex> shaderLock(*shaderMutex);

    // Double-check cache - another thread may have analyzed this shader while we waited
    if (s_cachingEnabled) {
      std::lock_guard<std::mutex> cacheLock(s_cacheMutex);
      auto it = s_analysisCache.find(hash);
      if (it != s_analysisCache.end()) {
        Logger::debug("[HlslDecompiler] Using cached analysis result (found while waiting for shader lock)");
        return it->second;
      }
    }

    // Write bytecode to temporary file
    std::string tempFile = writeBytecodeToTempFile(bytecode, bytecodeLength);
    if (tempFile.empty()) {
      result.errorMessage = "Failed to write bytecode to temporary file";
      return result;
    }

    std::string tempDir = std::filesystem::path(tempFile).parent_path().string();
    std::string baseName = std::filesystem::path(tempFile).stem().string();

    // Execute decompiler with semaphore to limit concurrent processes
    Logger::info(str::format("[HlslDecompiler] BEFORE SEMAPHORE - About to execute decompiler for shader hash: ", hash));

    // Acquire semaphore slot (wait if too many decompilers are running)
    {
      Logger::info(str::format("[HlslDecompiler] WAITING for semaphore slot (active: ", s_activeDecompilers, "/", s_maxConcurrentDecompilers, ")"));
      std::unique_lock<std::mutex> semLock(s_semaphoreMutex);
      s_semaphoreCV.wait(semLock, []() {
        return s_activeDecompilers < s_maxConcurrentDecompilers;
      });
      s_activeDecompilers++;
      Logger::info(str::format("[HlslDecompiler] ACQUIRED semaphore slot (", s_activeDecompilers, "/", s_maxConcurrentDecompilers, " now active)"));
    }

    Logger::info(str::format("[HlslDecompiler] BEFORE EXECUTE - tempFile: ", tempFile));

    bool decompilerSuccess = false;
    try {
      Logger::info("[HlslDecompiler] CALLING executeDecompiler...");
      decompilerSuccess = executeDecompiler(tempFile, tempDir, true);
      Logger::info(str::format("[HlslDecompiler] RETURNED from executeDecompiler, success=", decompilerSuccess));
    } catch (...) {
      Logger::err("[HlslDecompiler] EXCEPTION during decompiler execution");
      result.errorMessage = "Exception during decompiler execution";

      // Release semaphore slot
      {
        std::lock_guard<std::mutex> semLock(s_semaphoreMutex);
        s_activeDecompilers--;
        s_semaphoreCV.notify_one();
      }

      // Cache the failure so we don't retry
      if (s_cachingEnabled) {
        std::lock_guard<std::mutex> cacheLock(s_cacheMutex);
        s_analysisCache[hash] = result;
      }
      return result;
    }

    // Release semaphore slot after decompiler finishes
    Logger::info("[HlslDecompiler] >>> RELEASING semaphore slot...");
    {
      std::lock_guard<std::mutex> semLock(s_semaphoreMutex);
      s_activeDecompilers--;
      Logger::info(str::format("[HlslDecompiler] >>> RELEASED semaphore slot (", s_activeDecompilers, "/", s_maxConcurrentDecompilers, " now active)"));
      s_semaphoreCV.notify_one();
      Logger::info("[HlslDecompiler] >>> Notified waiting threads");
    }
    Logger::info("[HlslDecompiler] >>> Semaphore release complete");

    if (!decompilerSuccess) {
      result.errorMessage = "Failed to execute decompiler";
      Logger::warn(str::format("[HlslDecompiler] Decompiler execution failed for: ", tempFile));
      // Cache the failure so we don't retry
      if (s_cachingEnabled) {
        std::lock_guard<std::mutex> cacheLock(s_cacheMutex);
        s_analysisCache[hash] = result;
      }
      return result;
    }
    Logger::debug("[HlslDecompiler] Decompiler executed successfully");

    // Read decompiled output
    std::string hlslFile = tempDir + "/" + baseName + ".fx";
    std::string asmFile = tempDir + "/" + baseName + ".asm";

    Logger::debug(str::format("[HlslDecompiler] Checking for output files - HLSL: ", hlslFile));
    Logger::debug(str::format("[HlslDecompiler] Checking for output files - ASM: ", asmFile));

    bool hlslExists = std::filesystem::exists(hlslFile);
    bool asmExists = std::filesystem::exists(asmFile);

    Logger::debug(str::format("[HlslDecompiler] HLSL file exists: ", hlslExists ? "YES" : "NO"));
    Logger::debug(str::format("[HlslDecompiler] ASM file exists: ", asmExists ? "YES" : "NO"));

    result.decompiledHLSL = readFileContents(hlslFile);
    result.decompiledASM = readFileContents(asmFile);

    Logger::debug(str::format("[HlslDecompiler] HLSL content length: ", result.decompiledHLSL.length()));
    Logger::debug(str::format("[HlslDecompiler] ASM content length: ", result.decompiledASM.length()));

    if (result.decompiledHLSL.empty()) {
      result.errorMessage = "Failed to read decompiled HLSL output";
      Logger::warn("[HlslDecompiler] HLSL file not found or empty: " + hlslFile);
      return result;
    }

    // Parse shader type and model from ASM
    std::regex shaderHeaderPattern(R"((vs|ps)_(\d+)_(\d+))");
    std::smatch headerMatch;
    if (std::regex_search(result.decompiledASM, headerMatch, shaderHeaderPattern)) {
      result.shaderType = headerMatch[1].str();
      result.shaderModel = headerMatch[2].str() + "_" + headerMatch[3].str();
    }

    // Analyze matrices (with safety checks)
    try {
      Logger::debug("[HlslDecompiler] Starting parseMatrixUsage...");
      result.matrixInfo = parseMatrixUsage(result.decompiledHLSL, result.decompiledASM);
      Logger::debug("[HlslDecompiler] parseMatrixUsage complete");
    } catch (const std::exception& e) {
      Logger::err(str::format("[HlslDecompiler] Exception in parseMatrixUsage: ", e.what()));
    } catch (...) {
      Logger::err("[HlslDecompiler] Unknown exception in parseMatrixUsage");
    }

    // Analyze texture coordinates
    // IMPORTANT: Vertex shaders CAN output transformed TEXCOORDs that need neutralization!
    // Example: o.texcoord = dot(i.texcoord, vs_UVMatrix) causes sliding textures
    Logger::debug(str::format("[HlslDecompiler] Starting texture coordinate analysis for ", result.shaderType, " shader..."));

    try {
      result.texCoordInfo = parseTexCoordUsage(result.decompiledHLSL, result.decompiledASM);
      Logger::debug("[HlslDecompiler] parseTexCoordUsage complete");
    } catch (const std::exception& e) {
      Logger::err(str::format("[HlslDecompiler] Exception in parseTexCoordUsage: ", e.what()));
    } catch (...) {
      Logger::err("[HlslDecompiler] Unknown exception in parseTexCoordUsage");
    }

    // Only parse samplers for pixel shaders (vertex shaders don't sample textures)
    if (result.shaderType == "ps") {
      try {
        Logger::debug("[HlslDecompiler] Starting parseSamplerUsage...");
        result.samplers = parseSamplerUsage(result.decompiledHLSL);
        Logger::debug("[HlslDecompiler] parseSamplerUsage complete");
      } catch (const std::exception& e) {
        Logger::err(str::format("[HlslDecompiler] Exception in parseSamplerUsage: ", e.what()));
      } catch (...) {
        Logger::err("[HlslDecompiler] Unknown exception in parseSamplerUsage");
      }
    } else {
      Logger::debug("[HlslDecompiler] Skipping sampler analysis for vertex shader (but analyzing texcoords)");
    }

    // Determine recommended albedo sampler index
    result.albedoSamplerIndex = -1;
    int firstNonTechnicalSampler = -1;
    int currentSamplerIndex = 0;

    for (const auto& sampler : result.samplers) {
      // Determine the actual sampler index (explicit register or sequential assignment)
      int actualSamplerIndex = (sampler.registerIndex >= 0) ? sampler.registerIndex : currentSamplerIndex;

      if (!sampler.isTechnicalTexture) {
        // Track first non-technical sampler as fallback
        if (firstNonTechnicalSampler == -1) {
          firstNonTechnicalSampler = actualSamplerIndex;
        }

        // Prefer explicitly named albedo textures
        if (sampler.isProbablyAlbedo) {
          result.albedoSamplerIndex = actualSamplerIndex;
          break;  // Found a good match
        }
      }

      // Increment sequential sampler index for next sampler without explicit register
      if (sampler.registerIndex < 0) {
        currentSamplerIndex++;
      }
    }

    // Fallback to first non-technical sampler
    if (result.albedoSamplerIndex == -1) {
      result.albedoSamplerIndex = firstNonTechnicalSampler;
    }

    // Debug log sampler detection details
    if (!result.samplers.empty()) {
      std::string samplerNames;
      for (size_t i = 0; i < result.samplers.size(); i++) {
        const auto& samp = result.samplers[i];
        if (i > 0) samplerNames += ", ";
        samplerNames += samp.name;
        if (samp.registerIndex >= 0) {
          samplerNames += ":s" + std::to_string(samp.registerIndex);
        }
        if (samp.isTechnicalTexture) {
          samplerNames += "(tech)";
        }
        if (samp.isProbablyAlbedo) {
          samplerNames += "(albedo)";
        }
      }
      Logger::debug(str::format("[HlslDecompiler] Samplers: ", samplerNames));
    }

    result.success = true;

    // Cache result
    if (s_cachingEnabled) {
      std::lock_guard<std::mutex> cacheLock(s_cacheMutex);
      s_analysisCache[hash] = result;
    }

    // Log analysis results
    Logger::info(str::format(
      "[HlslDecompiler] Shader analysis complete: ",
      result.shaderType, "_", result.shaderModel,
      " | View=c", result.matrixInfo.viewMatrixRegister,
      " | Proj=c", result.matrixInfo.projectionMatrixRegister,
      " | World=c", result.matrixInfo.worldMatrixRegister,
      " | TexCoords=", result.texCoordInfo.texCoords.size(),
      " | Albedo=s", result.albedoSamplerIndex,
      " | Samplers=", result.samplers.size()
    ));

    return result;
  }

  std::string HlslDecompilerBridge::decompileToHLSL(const void* bytecode, size_t bytecodeLength) {
    ShaderAnalysisResult result = analyzeShader(bytecode, bytecodeLength);
    return result.decompiledHLSL;
  }

  std::string HlslDecompilerBridge::decompileToASM(const void* bytecode, size_t bytecodeLength) {
    ShaderAnalysisResult result = analyzeShader(bytecode, bytecodeLength);
    return result.decompiledASM;
  }

} // namespace dxvk
