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
#include "rtx/pass/shader_capture/shader_capture.h"

// Input: Capture requests from CPU
layout(binding = SHADER_CAPTURE_PREPARE_REQUESTS_INPUT)
StructuredBuffer<GpuCaptureRequest> CaptureRequests;

// Output: Indirect draw args for non-indexed draws
layout(binding = SHADER_CAPTURE_PREPARE_DRAW_ARGS_OUTPUT)
RWStructuredBuffer<IndirectDrawArgs> DrawArgs;

// Output: Indirect indexed draw args for indexed draws
layout(binding = SHADER_CAPTURE_PREPARE_INDEXED_ARGS_OUTPUT)
RWStructuredBuffer<IndirectIndexedDrawArgs> IndexedDrawArgs;

// GPU Counters for tracking
layout(binding = SHADER_CAPTURE_PREPARE_COUNTERS)
RWStructuredBuffer<GpuCaptureCounters> Counters;

// Push constants
layout(push_constant)
ConstantBuffer<ShaderCapturePrepareArgs> cb;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  uint requestIndex = threadId.x;

  // Bounds check
  if (requestIndex >= cb.numRequests)
  {
    return;
  }

  // Load capture request
  GpuCaptureRequest request = CaptureRequests[requestIndex];

  // Check if this is an indexed or non-indexed draw
  bool isIndexed = (request.indexCount > 0);

  if (isIndexed)
  {
    // Fill indexed draw args
    IndirectIndexedDrawArgs args;
    args.indexCount = request.indexCount;
    args.instanceCount = 1;
    args.firstIndex = request.indexOffset;
    args.vertexOffset = int(request.vertexOffset);
    args.firstInstance = requestIndex; // Use as draw ID for shader to identify which request

    IndexedDrawArgs[requestIndex] = args;
  }
  else
  {
    // Fill non-indexed draw args
    IndirectDrawArgs args;
    args.vertexCount = request.vertexCount;
    args.instanceCount = 1;
    args.firstVertex = request.vertexOffset;
    args.firstInstance = requestIndex; // Use as draw ID

    DrawArgs[requestIndex] = args;
  }

  // Atomic increment of processed count
  uint originalCount;
  InterlockedAdd(Counters[0].processedCaptures, 1, originalCount);
}
