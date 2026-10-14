// Copyright 2026 Divjot

#include "src/metrics.h"

#include <cmath>
#include <cstdlib>

namespace undistort {

double Psnr(const uint8_t* a, const uint8_t* b, size_t count) {
  double sse = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    sse += d * d;
  }
  if (sse == 0.0) return 99.0;
  const double mse = sse / static_cast<double>(count);
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

DiffStats Compare(const uint8_t* a, const uint8_t* b, size_t count) {
  DiffStats stats;
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
    if (d > stats.max_abs_diff) stats.max_abs_diff = d;
    if (d > 1) ++stats.pixels_over_1;
    sum += d;
  }
  stats.mean_abs_diff = count ? sum / static_cast<double>(count) : 0.0;
  return stats;
}

}  // namespace undistort
