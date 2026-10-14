// Copyright 2026 Divjot

#include "src/undistort_cpu.h"

#include <vector>

namespace undistort {

void BuildMapCpu(const UndistortParams& params, int width, int height,
                 std::vector<float>* map_x, std::vector<float>* map_y) {
  const size_t n = static_cast<size_t>(width) * height;
  map_x->resize(n);
  map_y->resize(n);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const size_t i = static_cast<size_t>(v) * width + u;
      MapOutputToSource(params, static_cast<float>(u), static_cast<float>(v),
                        &(*map_x)[i], &(*map_y)[i]);
    }
  }
}

void RemapCpu(const uint8_t* src, int width, int height, const float* map_x,
              const float* map_y, Interp interp, uint8_t* dst) {
  const int pitch = width * 3;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const size_t i = static_cast<size_t>(v) * width + u;
      SampleRgb(src, width, height, pitch, map_x[i], map_y[i], interp,
                dst + i * 3);
    }
  }
}

}  // namespace undistort
