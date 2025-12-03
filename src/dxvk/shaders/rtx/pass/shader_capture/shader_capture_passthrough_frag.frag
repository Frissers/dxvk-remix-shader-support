#version 450

// PASSTHROUGH FRAGMENT SHADER - No constant buffer binding required
// Simply samples texture and applies vertex color

// Input from vertex shader
layout(location = 0) in vec2 i_texcoord;
layout(location = 1) in vec4 i_color;

// Output color
layout(location = 0) out vec4 o_color;

// Texture sampler - uses D3D9 texture slot 0
// In DXVK, D3D9 samplers are bound starting at a computed slot
// We bind the game's texture at binding 0 in set 0
layout(set = 0, binding = 0) uniform sampler2D u_texture;

void main() {
  // Sample texture and output directly
  // Don't use vertex color - many game meshes don't provide it
  o_color = texture(u_texture, i_texcoord);
}
