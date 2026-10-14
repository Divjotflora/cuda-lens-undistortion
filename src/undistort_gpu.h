// Copyright 2026 Divjot
//
// GPU back ends for lens undistortion:
//   * kKernelMap   - custom kernel that builds a float map once per image size,
//                    plus a custom bilinear/nearest remap kernel per frame.
//   * kKernelFused - one custom kernel that evaluates the lens model and
//                    samples in the same thread (no map memory traffic).
//   * kNpp         - the same precomputed map fed to NPP's nppiRemap_8u_C3R.

#ifndef SRC_UNDISTORT_GPU_H_
#define SRC_UNDISTORT_GPU_H_

#include <cuda_runtime.h>
#include <npp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/camera_model.h"

#define UD_CUDA_CHECK(call)                                               \
  do {                                                                    \
    cudaError_t ud_err_ = (call);                                         \
    if (ud_err_ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                   cudaGetErrorName(ud_err_), __FILE__, __LINE__,         \
                   cudaGetErrorString(ud_err_));                          \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

#define UD_NPP_CHECK(call)                                                \
  do {                                                                    \
    NppStatus ud_status_ = (call);                                        \
    if (ud_status_ != NPP_SUCCESS) {                                      \
      std::fprintf(stderr, "NPP error %d at %s:%d\n",                     \
                   static_cast<int>(ud_status_), __FILE__, __LINE__);     \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

namespace undistort {

// Fills device maps (width*height floats each) on |stream|.
void LaunchBuildMap(const UndistortParams& params, int width, int height,
                    float* d_map_x, float* d_map_y, cudaStream_t stream);

// Custom remap kernel using precomputed maps.
void LaunchRemap(const uint8_t* d_src, int width, int height,
                 const float* d_map_x, const float* d_map_y, Interp interp,
                 uint8_t* d_dst, cudaStream_t stream);

// Custom fused kernel: lens model + sampling, no maps.
void LaunchFusedUndistort(const uint8_t* d_src, int width, int height,
                          const UndistortParams& params, Interp interp,
                          uint8_t* d_dst, cudaStream_t stream);

// Builds an NPP stream context for |stream| on the current device.
NppStreamContext MakeNppContext(cudaStream_t stream);

// NPP remap using the precomputed maps. Pixels whose source falls outside the
// image are left untouched by NPP, so |d_dst| is cleared first.
void NppRemap(const uint8_t* d_src, int width, int height,
              const float* d_map_x, const float* d_map_y, Interp interp,
              uint8_t* d_dst, const NppStreamContext& ctx);

}  // namespace undistort

#endif  // SRC_UNDISTORT_GPU_H_
