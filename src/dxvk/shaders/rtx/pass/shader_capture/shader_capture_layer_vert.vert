#version 450

#extension GL_ARB_shader_viewport_layer_array : enable

// Vertex inputs - only position and texcoord (no color - format mismatch issues)
// The C++ code binds: position at binding 0, texcoord at binding 1
// Position is vec3 - game vertex data is object space, we transform to clip space
layout(location = 0) in vec3 i_position;
layout(location = 1) in vec2 i_texcoord;
// NOTE: Removed i_color0 at location 2 - D3D9 games use R8G8B8A8_UINT which doesn't
// match vec4 float. We'll use a constant white color instead.

// Output to ALL possible locations (0-7) that the original PS might read from
// D3D9 assigns locations via RegisterLinkerSlot - we don't know which one at compile time
// So we output to ALL locations and let the PS read from whichever it needs
layout(location = 0) out vec4 o_slot0;
layout(location = 1) out vec4 o_slot1;
layout(location = 2) out vec4 o_slot2;
layout(location = 3) out vec4 o_slot3;
layout(location = 4) out vec4 o_slot4;
layout(location = 5) out vec4 o_slot5;
layout(location = 6) out vec4 o_slot6;
layout(location = 7) out vec4 o_slot7;

// Uniforms - push constants for projection matrix
layout(push_constant) uniform PushConstants {
  mat4 u_projection;
} constants;

// Custom vertex shader for shader captures with layer routing
// - Uses our projection for correct rendering
// - Routes gl_InstanceIndex to gl_Layer for batching
// - Outputs to ALL locations so original game PS can read what it needs
void main() {
  // Transform vertex to clip space using our projection
  // i_position is object space vec3, W=1.0 for position transformation
  gl_Position = constants.u_projection * vec4(i_position, 1.0);

  // Route instance index to layer for texture array rendering (batching)
  gl_Layer = gl_InstanceIndex;

  // Output texcoord and constant white color to ALL slots
  vec4 tc = vec4(i_texcoord, 0.0, 1.0);
  vec4 col = vec4(1.0, 1.0, 1.0, 1.0);  // Constant white - no color input

  // Output to ALL locations - PS will pick what it needs
  o_slot0 = tc;      // TEXCOORD0 often at slot 0
  o_slot1 = col;     // COLOR0 often at slot 1
  o_slot2 = tc;      // Fallback texcoord
  o_slot3 = col;     // Fallback color
  o_slot4 = tc;
  o_slot5 = col;
  o_slot6 = tc;
  o_slot7 = col;
}
