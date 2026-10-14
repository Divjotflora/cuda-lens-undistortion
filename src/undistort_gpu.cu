// Copyright 2026 Divjot

#include "src/undistort_gpu.h"

namespace undistort {
namespace {

// 32 x 8 threads: a warp covers 32 consecutive pixels of one row, so map
// loads (4 bytes per thread) and pixel stores are coalesced.
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

dim3 GridFor(int width, int height) {
  return dim3((width + kBlockX - 1) / kBlockX,
              (height + kBlockY - 1) / kBlockY);
}

__global__ void BuildMapKernel(UndistortParams params, int width, int height,
                               float* __restrict__ map_x,
                               float* __restrict__ map_y) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= width || v >= height) return;
  const int i = v * width + u;
  float sx, sy;
  MapOutputToSource(params, static_cast<float>(u), static_cast<float>(v), &sx,
                    &sy);
  map_x[i] = sx;
  map_y[i] = sy;
}

template <Interp kInterp>
__global__ void RemapKernel(const uint8_t* __restrict__ src, int width,
                            int height, const float* __restrict__ map_x,
                            const float* __restrict__ map_y,
                            uint8_t* __restrict__ dst) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= width || v >= height) return;
  const int i = v * width + u;
  SampleRgb(src, width, height, width * 3, map_x[i], map_y[i], kInterp,
            dst + 3 * i);
}

template <Interp kInterp>
__global__ void FusedUndistortKernel(const uint8_t* __restrict__ src,
                                     int width, int height,
                                     UndistortParams params,
                                     uint8_t* __restrict__ dst) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= width || v >= height) return;
  float sx, sy;
  MapOutputToSource(params, static_cast<float>(u), static_cast<float>(v), &sx,
                    &sy);
  SampleRgb(src, width, height, width * 3, sx, sy, kInterp,
            dst + 3 * (v * width + u));
}

}  // namespace

void LaunchBuildMap(const UndistortParams& params, int width, int height,
                    float* d_map_x, float* d_map_y, cudaStream_t stream) {
  BuildMapKernel<<<GridFor(width, height), dim3(kBlockX, kBlockY), 0,
                   stream>>>(params, width, height, d_map_x, d_map_y);
  UD_CUDA_CHECK(cudaGetLastError());
}

void LaunchRemap(const uint8_t* d_src, int width, int height,
                 const float* d_map_x, const float* d_map_y, Interp interp,
                 uint8_t* d_dst, cudaStream_t stream) {
  const dim3 grid = GridFor(width, height);
  const dim3 block(kBlockX, kBlockY);
  if (interp == Interp::kBilinear) {
    RemapKernel<Interp::kBilinear><<<grid, block, 0, stream>>>(
        d_src, width, height, d_map_x, d_map_y, d_dst);
  } else {
    RemapKernel<Interp::kNearest><<<grid, block, 0, stream>>>(
        d_src, width, height, d_map_x, d_map_y, d_dst);
  }
  UD_CUDA_CHECK(cudaGetLastError());
}

void LaunchFusedUndistort(const uint8_t* d_src, int width, int height,
                          const UndistortParams& params, Interp interp,
                          uint8_t* d_dst, cudaStream_t stream) {
  const dim3 grid = GridFor(width, height);
  const dim3 block(kBlockX, kBlockY);
  if (interp == Interp::kBilinear) {
    FusedUndistortKernel<Interp::kBilinear><<<grid, block, 0, stream>>>(
        d_src, width, height, params, d_dst);
  } else {
    FusedUndistortKernel<Interp::kNearest><<<grid, block, 0, stream>>>(
        d_src, width, height, params, d_dst);
  }
  UD_CUDA_CHECK(cudaGetLastError());
}

NppStreamContext MakeNppContext(cudaStream_t stream) {
  NppStreamContext ctx = {};
  int device = 0;
  UD_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  UD_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  ctx.hStream = stream;
  ctx.nCudaDeviceId = device;
  ctx.nMultiProcessorCount = prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
  ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  UD_CUDA_CHECK(cudaStreamGetFlags(stream, &ctx.nStreamFlags));
  return ctx;
}

void NppRemap(const uint8_t* d_src, int width, int height,
              const float* d_map_x, const float* d_map_y, Interp interp,
              uint8_t* d_dst, const NppStreamContext& ctx) {
  const int step = width * 3;
  const int map_step = width * static_cast<int>(sizeof(float));
  const NppiSize size = {width, height};
  const NppiRect roi = {0, 0, width, height};
  UD_CUDA_CHECK(cudaMemsetAsync(d_dst, 0, static_cast<size_t>(step) * height,
                                ctx.hStream));
  const int npp_interp =
      interp == Interp::kBilinear ? NPPI_INTER_LINEAR : NPPI_INTER_NN;
  UD_NPP_CHECK(nppiRemap_8u_C3R_Ctx(d_src, size, step, roi, d_map_x, map_step,
                                    d_map_y, map_step, d_dst, step, size,
                                    npp_interp, ctx));
}

}  // namespace undistort
