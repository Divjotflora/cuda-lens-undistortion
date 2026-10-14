// Copyright 2026 Divjot
//
// Brown-Conrady lens model shared by the CPU reference and the CUDA kernels.
// Every function that does per-pixel math is marked host+device so the GPU
// and CPU code paths execute exactly the same arithmetic.

#ifndef SRC_CAMERA_MODEL_H_
#define SRC_CAMERA_MODEL_H_

#include <cmath>
#include <cstdint>
#include <string>

#ifdef __CUDACC__
#define UD_HOST_DEVICE __host__ __device__
#else
#define UD_HOST_DEVICE
#endif

namespace undistort {

// Lens description that is independent of image resolution. The focal length
// is expressed as a fraction of the image width so one config works for any
// frame size produced by the same lens.
struct LensConfig {
  float focal_scale = 0.6f;  // fx = fy = focal_scale * width
  float k1 = 0.0f;           // radial coefficients
  float k2 = 0.0f;
  float k3 = 0.0f;
  float p1 = 0.0f;           // tangential coefficients
  float p2 = 0.0f;
};

// Fully resolved parameters for one image size. "in_*" are the intrinsics of
// the distorted input; "out_*" are the intrinsics of the corrected output.
struct UndistortParams {
  float in_fx, in_fy, in_cx, in_cy;
  float out_fx, out_fy, out_cx, out_cy;
  float k1, k2, k3, p1, p2;
};

enum class Interp { kNearest, kBilinear };

// Builds per-resolution parameters. |zoom| > 1 crops in, < 1 shows more of the
// (curved) border of the distorted frame.
UndistortParams MakeParams(const LensConfig& lens, int width, int height,
                           float zoom);

// Parses a "key=value" text file (keys: focal_scale, k1, k2, k3, p1, p2).
bool LoadLensConfig(const std::string& path, LensConfig* lens,
                    std::string* error);

// Sets one lens field by name. Returns false on an unknown key.
bool SetLensField(const std::string& key, float value, LensConfig* lens);

std::string LensToString(const LensConfig& lens);

// For an output (undistorted) pixel (u, v) returns the location (su, sv) in
// the distorted input that must be sampled. This is the forward lens model,
// which is why undistortion needs no iterative inversion.
UD_HOST_DEVICE inline void MapOutputToSource(const UndistortParams& p, float u,
                                             float v, float* su, float* sv) {
  const float x = (u - p.out_cx) / p.out_fx;
  const float y = (v - p.out_cy) / p.out_fy;
  const float r2 = x * x + y * y;
  const float radial = 1.0f + r2 * (p.k1 + r2 * (p.k2 + r2 * p.k3));
  const float xd =
      x * radial + 2.0f * p.p1 * x * y + p.p2 * (r2 + 2.0f * x * x);
  const float yd =
      y * radial + p.p1 * (r2 + 2.0f * y * y) + 2.0f * p.p2 * x * y;
  *su = p.in_fx * xd + p.in_cx;
  *sv = p.in_fy * yd + p.in_cy;
}

// Samples an interleaved RGB8 image. Pixel centres sit on integer
// coordinates; samples outside [0, w-1] x [0, h-1] produce black.
UD_HOST_DEVICE inline void SampleRgb(const uint8_t* src, int width, int height,
                                     int pitch, float sx, float sy,
                                     Interp interp, uint8_t* out) {
  if (!(sx >= 0.0f && sy >= 0.0f && sx <= width - 1.0f &&
        sy <= height - 1.0f)) {
    out[0] = out[1] = out[2] = 0;
    return;
  }
  if (interp == Interp::kNearest) {
    const int x = static_cast<int>(sx + 0.5f);
    const int y = static_cast<int>(sy + 0.5f);
    const uint8_t* px = src + y * pitch + x * 3;
    out[0] = px[0];
    out[1] = px[1];
    out[2] = px[2];
    return;
  }
  const int x0 = static_cast<int>(sx);
  const int y0 = static_cast<int>(sy);
  const int x1 = x0 + 1 < width ? x0 + 1 : x0;
  const int y1 = y0 + 1 < height ? y0 + 1 : y0;
  const float fx = sx - x0;
  const float fy = sy - y0;
  const uint8_t* r0 = src + y0 * pitch;
  const uint8_t* r1 = src + y1 * pitch;
  for (int c = 0; c < 3; ++c) {
    const float top = r0[x0 * 3 + c] + fx * (r0[x1 * 3 + c] - r0[x0 * 3 + c]);
    const float bot = r1[x0 * 3 + c] + fx * (r1[x1 * 3 + c] - r1[x0 * 3 + c]);
    const float val = top + fy * (bot - top) + 0.5f;
    out[c] = static_cast<uint8_t>(val > 255.0f ? 255.0f : val);
  }
}

}  // namespace undistort

#endif  // SRC_CAMERA_MODEL_H_
