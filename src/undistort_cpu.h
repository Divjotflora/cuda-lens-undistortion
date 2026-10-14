// Copyright 2026 Divjot
//
// Single-threaded CPU reference implementation. It is used to verify the GPU
// output pixel-for-pixel and as the baseline for the reported speed-up.

#ifndef SRC_UNDISTORT_CPU_H_
#define SRC_UNDISTORT_CPU_H_

#include <cstdint>
#include <vector>

#include "src/camera_model.h"

namespace undistort {

// Precomputes the source-coordinate maps (structure of arrays).
void BuildMapCpu(const UndistortParams& params, int width, int height,
                 std::vector<float>* map_x, std::vector<float>* map_y);

// Applies precomputed maps to an interleaved RGB8 image.
void RemapCpu(const uint8_t* src, int width, int height, const float* map_x,
              const float* map_y, Interp interp, uint8_t* dst);

}  // namespace undistort

#endif  // SRC_UNDISTORT_CPU_H_
