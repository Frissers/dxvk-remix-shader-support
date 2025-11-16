/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef SHADER_CAPTURE_H
#define SHADER_CAPTURE_H

#include "rtx/utility/shader_types.h"

// Binding indices for shader capture prepare compute shader
#define SHADER_CAPTURE_PREPARE_REQUESTS_INPUT    0
#define SHADER_CAPTURE_PREPARE_DRAW_ARGS_OUTPUT  1
#define SHADER_CAPTURE_PREPARE_INDEXED_ARGS_OUTPUT 2
#define SHADER_CAPTURE_PREPARE_COUNTERS          3

// GPU Capture Request - matches C++ struct in rtx_shader_output_capturer.h
struct GpuCaptureRequest {
  uint64_t materialHash;
  uint64_t geometryHash;
  uint drawCallIndex;
  uint renderTargetIndex;
  uint vertexOffset;
  uint vertexCount;
  uint indexOffset;
  uint indexCount;
  uint2 resolution;
  uint flags;
  uint padding[2];
};

// Indirect Draw Args - VkDrawIndirectCommand
struct IndirectDrawArgs {
  uint vertexCount;
  uint instanceCount;
  uint firstVertex;
  uint firstInstance;
};

// Indirect Indexed Draw Args - VkDrawIndexedIndirectCommand
struct IndirectIndexedDrawArgs {
  uint indexCount;
  uint instanceCount;
  uint firstIndex;
  int  vertexOffset;
  uint firstInstance;
};

// GPU Counters
struct GpuCaptureCounters {
  uint totalCaptureRequests;
  uint processedCaptures;
  uint skippedCaptures;
  uint failedCaptures;
  uint padding[12];
};

// Push constants for prepare shader
struct ShaderCapturePrepareArgs {
  uint numRequests;
  uint maxCapturesPerFrame;
  uint padding[2];
};

#endif // SHADER_CAPTURE_H
