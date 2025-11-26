/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* RTX Remix HLSL Decompiler Bridge
* This file provides integration with HlslDecompiler for automatic shader analysis
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <cstdint>

namespace dxvk {

  // Shader matrix usage information
  struct ShaderMatrixInfo {
    int worldMatrixRegister = -1;
    int viewMatrixRegister = -1;
    int projectionMatrixRegister = -1;
    int worldViewMatrixRegister = -1;
    int worldViewProjMatrixRegister = -1;
    bool isRowMajor = false;
    bool isColumnMajor = false;

    // Additional matrix registers that might be present
    std::vector<int> otherMatrixRegisters;
  };

  // Sampler information
  struct ShaderSamplerInfo {
    std::string name;                   // Sampler name (e.g., "g_AlbedoTex", "texture0")
    int registerIndex = -1;             // Sampler register (s0, s1, etc.)
    bool isTechnicalTexture = false;    // Is this a depth/blur/effect texture?
    bool isProbablyAlbedo = false;      // Likely the main diffuse/albedo texture
  };

  // Texture coordinate usage information
  struct ShaderTexCoordInfo {
    struct TexCoordUsage {
      int texCoordIndex = -1;           // Which texcoord (TEXCOORD0, TEXCOORD1, etc.)
      int samplerIndex = -1;            // Which sampler uses this texcoord
      bool isTransformed = false;       // Is it multiplied by a matrix?
      int transformMatrixRegister = -1; // If transformed, which register?
      bool isProjected = false;         // Is it a projected texture coordinate?
      std::vector<std::string> operations; // List of operations performed on this texcoord
    };

    std::vector<TexCoordUsage> texCoords;
    bool hasWorldSpaceTexCoords = false;
    bool hasViewSpaceTexCoords = false;
  };

  // Shader analysis result
  struct ShaderAnalysisResult {
    bool success = false;
    std::string errorMessage;

    ShaderMatrixInfo matrixInfo;
    ShaderTexCoordInfo texCoordInfo;
    std::vector<ShaderSamplerInfo> samplers;

    // Recommended sampler index for albedo texture (-1 if unknown)
    int albedoSamplerIndex = -1;

    // Decompiled HLSL source for inspection
    std::string decompiledHLSL;
    std::string decompiledASM;

    // Shader type and version
    std::string shaderType;  // "vs" or "ps"
    std::string shaderModel; // "3_0", "2_0", etc.
  };

  /**
   * \brief HLSL Decompiler Bridge
   *
   * This class provides integration with the HlslDecompiler tool to automatically
   * analyze shaders and extract matrix, texture coordinate, and other usage information.
   */
  class HlslDecompilerBridge {
  public:
    /**
     * \brief Analyzes a shader bytecode blob
     * \param bytecode Pointer to shader bytecode
     * \param bytecodeLength Length of shader bytecode
     * \returns Analysis result with matrix and texcoord information
     */
    static ShaderAnalysisResult analyzeShader(const void* bytecode, size_t bytecodeLength);

    /**
     * \brief Decompiles shader to HLSL
     * \param bytecode Pointer to shader bytecode
     * \param bytecodeLength Length of shader bytecode
     * \returns Decompiled HLSL source code
     */
    static std::string decompileToHLSL(const void* bytecode, size_t bytecodeLength);

    /**
     * \brief Decompiles shader to assembly
     * \param bytecode Pointer to shader bytecode
     * \param bytecodeLength Length of shader bytecode
     * \returns Decompiled assembly source code
     */
    static std::string decompileToASM(const void* bytecode, size_t bytecodeLength);

    /**
     * \brief Gets the path to the HlslDecompiler executable
     */
    static std::string getDecompilerPath();

    /**
     * \brief Sets whether to enable shader analysis caching
     */
    static void setCachingEnabled(bool enabled);

    /**
     * \brief Get number of shaders pending decompilation
     */
    static uint32_t getPendingDecompilationCount() { return s_pendingDecompilations.load(); }

    /**
     * \brief Get total number of shaders decompiled this session
     */
    static uint32_t getDecompiledCount() { return s_decompiledCount.load(); }

    /**
     * \brief Increment pending decompilation count (call before starting decompilation)
     */
    static void incrementPending() { s_pendingDecompilations++; }

    /**
     * \brief Decrement pending decompilation count (call after decompilation completes)
     */
    static void decrementPending() {
      s_pendingDecompilations--;
      s_decompiledCount++;
    }

  private:
    // Progress tracking for UI
    static std::atomic<uint32_t> s_pendingDecompilations;
    static std::atomic<uint32_t> s_decompiledCount;
    static bool s_cachingEnabled;
    static std::map<std::string, ShaderAnalysisResult> s_analysisCache;
    static std::mutex s_cacheMutex;  // Protects cache access
    static std::unordered_map<std::string, std::shared_ptr<std::mutex>> s_shaderMutexes;  // Per-shader locks
    static std::mutex s_mutexMapMutex;  // Protects the shader mutex map

    // Semaphore to limit concurrent decompiler processes (prevents overwhelming the system)
    static std::mutex s_semaphoreMutex;
    static std::condition_variable s_semaphoreCV;
    static int s_activeDecompilers;
    static const int s_maxConcurrentDecompilers = 8;  // Max 8 concurrent decompiler processes (good balance)

    /**
     * \brief Executes HlslDecompiler and captures output
     */
    static bool executeDecompiler(const std::string& inputFile, const std::string& outputDir, bool doAstAnalysis);

    /**
     * \brief Parses decompiled HLSL to extract matrix usage
     */
    static ShaderMatrixInfo parseMatrixUsage(const std::string& hlsl, const std::string& asm_code);

    /**
     * \brief Parses decompiled HLSL to extract texture coordinate usage
     */
    static ShaderTexCoordInfo parseTexCoordUsage(const std::string& hlsl, const std::string& asm_code);

    /**
     * \brief Parses decompiled HLSL to extract sampler information
     */
    static std::vector<ShaderSamplerInfo> parseSamplerUsage(const std::string& hlsl);

    /**
     * \brief Writes bytecode to temporary file
     */
    static std::string writeBytecodeToTempFile(const void* bytecode, size_t bytecodeLength);

    /**
     * \brief Reads file contents into string
     */
    static std::string readFileContents(const std::string& path);

    /**
     * \brief Computes hash of bytecode for caching
     */
    static std::string computeBytecodeHash(const void* bytecode, size_t bytecodeLength);
  };

} // namespace dxvk
