#version 450

// Fragment inputs from vertex shader
layout(location = 0) in vec2 i_texcoord;

// Fragment output
layout(location = 0) out vec4 o_color;

// Texture sampler - bound at set 0, binding 0
layout(set = 0, binding = 0) uniform sampler2D u_texture;

// Simple fragment shader for shader capture - just samples the texture
void main() {
  o_color = texture(u_texture, i_texcoord);
}
