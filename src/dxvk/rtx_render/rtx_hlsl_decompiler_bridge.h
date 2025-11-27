#pragma once

#include <string>
#include <cstdint>
#include <optional>
#include <memory>

namespace dxvk {

// Forward declarations
class ShaderDecompilationCache;

/**
 * \brief Bridge to HlslDecompiler for analyzing shader bytecode
 *
 * This class provides an interface to the external HlslDecompiler tool
 * which can decompile D3D9 shaders to extract transformation matrices,
 * texture sampling information, and other shader characteristics.
 */
class HlslDecompilerBridge {
public:
  /**
   * \brief Structure containing decompiled shader information
   */
  struct ShaderInfo {
    bool hasWorldMatrix = false;
    bool hasViewMatrix = false;
    bool hasProjectionMatrix = false;
    bool hasWorldViewMatrix = false;
    bool hasWorldViewProjMatrix = false;

    // Matrix transformation issues
    bool hasTransposeIssue = false;
    bool hasInverseIssue = false;

    // Texture sampling info
    int numTextureSamples = 0;
    bool hasTexcoordTransform = false;

    // Vertex shader specific
    bool outputsPosition = false;
    bool outputsNormal = false;
    bool outputsTexcoord = false;

    std::string decompiledHLSL;
  };

  HlslDecompilerBridge(const std::string& decompilerPath = "");
  ~HlslDecompilerBridge();

  /**
   * \brief Decompile shader bytecode and extract information
   *
   * \param bytecode Shader bytecode
   * \param bytecodeLength Length of bytecode in bytes
   * \param shaderHash Hash of the shader for caching
   * \returns Decompiled shader info if successful
   */
  std::optional<ShaderInfo> decompileShader(
    const void* bytecode,
    size_t bytecodeLength,
    uint64_t shaderHash
  );

  /**
   * \brief Set whether to enable caching
   */
  void setEnableCaching(bool enable) { m_enableCaching = enable; }

  /**
   * \brief Set verbose logging
   */
  void setVerbose(bool verbose) { m_verbose = verbose; }

private:
  std::string m_decompilerPath;
  std::unique_ptr<ShaderDecompilationCache> m_cache;
  bool m_enableCaching = true;
  bool m_verbose = false;

  bool executeDecompiler(const std::string& inputPath, const std::string& outputPath);
  std::optional<ShaderInfo> parseDecompiledHLSL(const std::string& hlslCode);
};

} // namespace dxvk
