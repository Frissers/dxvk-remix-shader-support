#version 450

#extension GL_ARB_shader_viewport_layer_array : enable

// Vertex inputs - only position and texcoord needed for shader capture
layout(location = 0) in vec3 i_position;
layout(location = 1) in vec2 i_texcoord;

// Vertex outputs
layout(location = 0) out vec2 o_texcoord;

// Uniforms
layout(push_constant) uniform PushConstants {
  mat4 u_projection;
} constants;

// Simple vertex shader for shader captures with instanced layer routing
// gl_InstanceIndex is available in vertex shaders, so we can output gl_Layer directly!
void main() {
  // Transform vertex to clip space
  gl_Position = constants.u_projection * vec4(i_position, 1.0);

  // Route instance index to layer for texture array rendering
  gl_Layer = gl_InstanceIndex;

  // Pass through texcoord to fragment shader
  o_texcoord = i_texcoord;
}
