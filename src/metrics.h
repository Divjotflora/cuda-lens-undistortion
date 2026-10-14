// Copyright 2026 Divjot
//
// Image-quality and correctness metrics.

#ifndef SRC_METRICS_H_
#define SRC_METRICS_H_

#include <cstddef>
#include <cstdint>

namespace undistort {

struct DiffStats {
  int max_abs_diff = 0;
  double mean_abs_diff = 0.0;
  size_t pixels_over_1 = 0;  // channel values that differ by more than 1
};

// Peak signal-to-noise ratio in dB over |count| bytes. Returns 99 for a
// perfect match so logs stay readable.
double Psnr(const uint8_t* a, const uint8_t* b, size_t count);

DiffStats Compare(const uint8_t* a, const uint8_t* b, size_t count);

}  // namespace undistort

#endif  // SRC_METRICS_H_
