#version 450

// PASSTHROUGH VERTEX SHADER - No constant buffer binding required
// Uses DXSO-compatible attribute locations to match game's vertex format
// Only uses push constants (64 bytes) for projection matrix

// DXSO vertex attribute locations (must match what DXSO compiler uses):
// POSITION/POSITIONT = 0
// BLENDWEIGHT = 1
// BLENDINDICES = 2
// NORMAL = 3
// COLOR0 = 4, COLOR1 = 5
// TEXCOORD0 = 7, TEXCOORD1 = 8, etc.

layout(location = 0) in vec4 i_position;    // POSITION (vec4: xyz + w from vertex capture)
layout(location = 7) in vec2 i_texcoord0;   // TEXCOORD0

// Output to fragment shader - MUST match DXSO locations for game PS compatibility!
// DXSO outputs: TEXCOORD0 = location 7, COLOR0 = location 4
layout(location = 7) out vec2 o_texcoord;

// Push constants - projection matrix (identity for clip-space, WVP for object-space)
layout(push_constant) uniform PushConstants {
  mat4 u_projection;
} pc;

void main() {
  // Check if projection matrix is identity (indicates clip-space positions from vertex capture
  // when RTX transforms were identity and couldn't invert)
  // Identity matrix: diagonal = 1, off-diagonal = 0
  bool isIdentity = (pc.u_projection[0][0] == 1.0 && pc.u_projection[1][1] == 1.0 &&
                     pc.u_projection[2][2] == 1.0 && pc.u_projection[3][3] == 1.0 &&
                     pc.u_projection[0][1] == 0.0 && pc.u_projection[0][2] == 0.0);

  if (isIdentity) {
    // Clip-space positions - use full vec4 including W for correct perspective
    // The W component was preserved from the original clip position
    gl_Position = i_position;
  } else {
    // Object-space positions - apply WVP transform with W=1.0
    gl_Position = pc.u_projection * vec4(i_position.xyz, 1.0);
  }

  // Pass through texcoord
  o_texcoord = i_texcoord0;
}
