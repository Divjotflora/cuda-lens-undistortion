// Copyright 2026 Divjot

#include "src/batch_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/camera_model.h"
#include "src/image_io.h"
#include "src/metrics.h"
#include "src/undistort_cpu.h"
#include "src/undistort_gpu.h"

namespace undistort {
namespace {

using Clock = std::chrono::steady_clock;
using SizeKey = std::pair<int, int>;

// GPU results may differ from the CPU reference by one rounding step; the
// lens model is evaluated with FMA contraction on the device.
constexpr int kKernelTolerance = 2;
constexpr double kNppMeanTolerance = 1.0;

double MillisSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct CpuMap {
  std::vector<float> x;
  std::vector<float> y;
};

// Caches CPU maps per resolution (reference path and verification).
class CpuMapCache {
 public:
  CpuMapCache(const LensConfig& lens, float zoom) : lens_(lens), zoom_(zoom) {}

  const CpuMap& Get(int width, int height) {
    const SizeKey key(width, height);
    auto it = maps_.find(key);
    if (it == maps_.end()) {
      CpuMap map;
      BuildMapCpu(MakeParams(lens_, width, height, zoom_), width, height,
                  &map.x, &map.y);
      it = maps_.emplace(key, std::move(map)).first;
    }
    return it->second;
  }

 private:
  LensConfig lens_;
  float zoom_;
  std::map<SizeKey, CpuMap> maps_;
};

// Collects per-image results: PSNR against ground truth, verification
// statistics, optional CSV and output files, and the final summary.
class ResultSink {
 public:
  ResultSink(const Options& options, size_t total)
      : options_(options), total_(total) {}

  bool Open(std::string* error) {
    if (options_.write_outputs && !MakeDirs(options_.output_dir, error)) {
      return false;
    }
    if (!options_.csv_path.empty()) {
      csv_.open(options_.csv_path);
      if (!csv_) {
        *error = "cannot write " + options_.csv_path;
        return false;
      }
      csv_ << "index,name,width,height,compute_ms,transfer_ms,psnr_input_db,"
              "psnr_output_db,verify_max_abs_diff\n";
    }
    return true;
  }

  void Add(int index, const ImageInfo& info, const uint8_t* input,
           const uint8_t* output, double compute_ms, double transfer_ms,
           const DiffStats* verify) {
    const size_t bytes = static_cast<size_t>(info.width) * info.height * 3;
    ++processed_;
    megapixels_ += info.width * static_cast<double>(info.height) / 1e6;
    compute_ms_ += compute_ms;
    transfer_ms_ += transfer_ms;
    sizes_.insert(SizeKey(info.width, info.height));

    double psnr_in = -1.0, psnr_out = -1.0;
    if (!options_.gt_dir.empty()) {
      std::vector<uint8_t> gt;
      int gw = 0, gh = 0;
      std::string error;
      const std::string gt_path = options_.gt_dir + "/" + info.name;
      if (ReadPpm(gt_path, &gt, &gw, &gh, &error) && gw == info.width &&
          gh == info.height) {
        psnr_in = Psnr(input, gt.data(), bytes);
        psnr_out = Psnr(output, gt.data(), bytes);
        psnr_in_sum_ += psnr_in;
        psnr_out_sum_ += psnr_out;
        ++psnr_count_;
      } else if (!warned_gt_) {
        std::printf("  [warn] no usable ground truth for %s (%s)\n",
                    info.name.c_str(), error.c_str());
        warned_gt_ = true;
      }
    }

    if (verify != nullptr) {
      ++verified_;
      worst_diff_ = std::max(worst_diff_, verify->max_abs_diff);
      worst_mean_diff_ = std::max(worst_mean_diff_, verify->mean_abs_diff);
    }

    if (options_.write_outputs) {
      std::string error;
      if (!WritePpm(options_.output_dir + "/" + info.name, output, info.width,
                    info.height, &error)) {
        std::printf("  [error] %s\n", error.c_str());
        ++errors_;
      }
    }

    if (csv_.is_open()) {
      csv_ << index << "," << info.name << "," << info.width << ","
           << info.height << "," << compute_ms << "," << transfer_ms << ","
           << psnr_in << "," << psnr_out << ","
           << (verify ? verify->max_abs_diff : -1) << "\n";
    }

    if (!options_.quiet) {
      std::printf("[%4d/%zu] %-22s %5dx%-5d compute %8.3f ms", index + 1,
                  total_, info.name.c_str(), info.width, info.height,
                  compute_ms);
      if (transfer_ms > 0.0) std::printf("  xfer %7.3f ms", transfer_ms);
      if (psnr_out >= 0.0) {
        std::printf("  PSNR %5.2f -> %5.2f dB", psnr_in, psnr_out);
      }
      if (verify) std::printf("  verify maxdiff=%d", verify->max_abs_diff);
      std::printf("\n");
    }
  }

  void AddError() { ++errors_; }
  void AddCpuReferenceTime(double cpu_ms, double gpu_ms) {
    cpu_ref_ms_ += cpu_ms;
    gpu_ref_ms_ += gpu_ms;
  }

  // Returns true when all verification checks passed.
  bool PrintSummary(double wall_ms, double map_ms) const {
    const bool is_gpu = options_.backend != Backend::kCpu;
    std::printf("\n================ Summary ================\n");
    std::printf("backend            : %s (%s)\n", BackendName(options_.backend),
                InterpName(options_.interp));
    std::printf("images processed   : %zu of %zu (%zu distinct size%s)\n",
                processed_, total_, sizes_.size(),
                sizes_.size() == 1 ? "" : "s");
    std::printf("errors             : %zu\n", errors_);
    std::printf("total megapixels   : %.2f MP\n", megapixels_);
    std::printf("wall time          : %.1f ms  (%.1f images/s, %.1f MP/s "
                "end-to-end incl. disk I/O)\n",
                wall_ms, processed_ * 1000.0 / wall_ms,
                megapixels_ * 1000.0 / wall_ms);
    if (map_ms > 0.0) {
      std::printf("map build          : %.3f ms (once per resolution)\n",
                  map_ms);
    }
    std::printf("compute time       : %.2f ms total, %.3f ms/image, %.1f "
                "MP/s\n",
                compute_ms_, processed_ ? compute_ms_ / processed_ : 0.0,
                compute_ms_ > 0 ? megapixels_ * 1000.0 / compute_ms_ : 0.0);
    if (is_gpu) {
      std::printf("H2D + D2H copies   : %.2f ms total\n", transfer_ms_);
    }
    if (psnr_count_ > 0) {
      std::printf("PSNR vs truth      : distorted input %.2f dB -> corrected "
                  "%.2f dB (mean of %zu)\n",
                  psnr_in_sum_ / psnr_count_, psnr_out_sum_ / psnr_count_,
                  psnr_count_);
    }
    bool ok = errors_ == 0;
    if (is_gpu && verified_ > 0) {
      bool pass;
      if (options_.backend == Backend::kNpp) {
        pass = worst_mean_diff_ <= kNppMeanTolerance;
        std::printf("verify vs CPU      : %zu images, max |diff| %d, worst "
                    "mean |diff| %.4f -> %s (NPP tolerance: mean <= %.1f)\n",
                    verified_, worst_diff_, worst_mean_diff_,
                    pass ? "PASS" : "WARN", kNppMeanTolerance);
      } else {
        pass = worst_diff_ <= kKernelTolerance;
        std::printf("verify vs CPU      : %zu images, max |diff| %d, worst "
                    "mean |diff| %.4f -> %s (tolerance %d)\n",
                    verified_, worst_diff_, worst_mean_diff_,
                    pass ? "PASS" : "FAIL", kKernelTolerance);
        ok = ok && pass;
      }
      if (gpu_ref_ms_ > 0.0) {
        std::printf("speed-up vs 1 CPU  : %.1fx (compute only, on the "
                    "verified images: CPU %.2f ms vs GPU %.3f ms)\n",
                    cpu_ref_ms_ / gpu_ref_ms_, cpu_ref_ms_ / verified_,
                    gpu_ref_ms_ / verified_);
      }
    }
    std::printf("result             : %s\n", ok ? "OK" : "FAILED");
    return ok;
  }

 private:
  const Options& options_;
  size_t total_;
  std::ofstream csv_;
  std::set<SizeKey> sizes_;
  size_t processed_ = 0, errors_ = 0, verified_ = 0, psnr_count_ = 0;
  double megapixels_ = 0.0, compute_ms_ = 0.0, transfer_ms_ = 0.0;
  double psnr_in_sum_ = 0.0, psnr_out_sum_ = 0.0;
  double cpu_ref_ms_ = 0.0, gpu_ref_ms_ = 0.0, worst_mean_diff_ = 0.0;
  int worst_diff_ = 0;
  bool warned_gt_ = false;
};

// ---------------------------------------------------------------- CPU path

int RunCpu(const Options& options, const std::vector<ImageInfo>& images) {
  ResultSink sink(options, images.size());
  std::string error;
  if (!sink.Open(&error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }
  CpuMapCache maps(options.lens, options.zoom);
  std::vector<uint8_t> input, output;
  double map_ms = 0.0;
  const auto wall_start = Clock::now();
  for (size_t i = 0; i < images.size(); ++i) {
    const ImageInfo& info = images[i];
    int w = 0, h = 0;
    if (!ReadPpm(info.path, &input, &w, &h, &error)) {
      std::printf("  [error] %s\n", error.c_str());
      sink.AddError();
      continue;
    }
    const auto map_start = Clock::now();
    const CpuMap& map = maps.Get(w, h);
    map_ms += MillisSince(map_start);
    output.resize(input.size());
    const auto start = Clock::now();
    RemapCpu(input.data(), w, h, map.x.data(), map.y.data(), options.interp,
             output.data());
    sink.Add(static_cast<int>(i), info, input.data(), output.data(),
             MillisSince(start), 0.0, nullptr);
  }
  return sink.PrintSummary(MillisSince(wall_start), map_ms) ? 0 : 1;
}

// ---------------------------------------------------------------- GPU path

struct DeviceMap {
  float* x = nullptr;
  float* y = nullptr;
};

struct StreamSlot {
  cudaStream_t stream = nullptr;
  cudaEvent_t ev_begin = nullptr;    // before H2D
  cudaEvent_t ev_kernel = nullptr;   // after H2D, before kernel
  cudaEvent_t ev_kdone = nullptr;    // after kernel
  cudaEvent_t ev_end = nullptr;      // after D2H
  NppStreamContext npp_ctx;
  uint8_t* h_in = nullptr;
  uint8_t* h_out = nullptr;
  uint8_t* d_in = nullptr;
  uint8_t* d_out = nullptr;
  int image_index = -1;  // image currently in flight, -1 when idle
};

void PrintDevice(int device) {
  cudaDeviceProp prop;
  UD_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime = 0, driver = 0;
  UD_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  UD_CUDA_CHECK(cudaDriverGetVersion(&driver));
  std::printf("GPU %d              : %s (sm_%d%d, %d SMs, %.1f GB)\n", device,
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.totalGlobalMem / 1073741824.0);
  std::printf("CUDA runtime/driver: %d / %d\n", runtime, driver);
}

int RunGpu(const Options& options, const std::vector<ImageInfo>& images) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                 "error: no CUDA device available (use --backend cpu to run "
                 "the reference implementation)\n");
    return 1;
  }
  if (options.device >= device_count) {
    std::fprintf(stderr, "error: --device %d but only %d device(s)\n",
                 options.device, device_count);
    return 1;
  }
  UD_CUDA_CHECK(cudaSetDevice(options.device));
  PrintDevice(options.device);

  ResultSink sink(options, images.size());
  std::string error;
  if (!sink.Open(&error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }

  size_t max_bytes = 0;
  std::set<SizeKey> sizes;
  for (const ImageInfo& info : images) {
    max_bytes = std::max(max_bytes,
                         static_cast<size_t>(info.width) * info.height * 3);
    sizes.insert(SizeKey(info.width, info.height));
  }

  // One map per resolution, built on the GPU and reused for every frame.
  std::map<SizeKey, DeviceMap> device_maps;
  float map_ms = 0.0f;
  if (options.backend != Backend::kKernelFused) {
    cudaEvent_t start, stop;
    UD_CUDA_CHECK(cudaEventCreate(&start));
    UD_CUDA_CHECK(cudaEventCreate(&stop));
    UD_CUDA_CHECK(cudaEventRecord(start));
    for (const SizeKey& size : sizes) {
      DeviceMap map;
      const size_t n = static_cast<size_t>(size.first) * size.second;
      UD_CUDA_CHECK(cudaMalloc(&map.x, n * sizeof(float)));
      UD_CUDA_CHECK(cudaMalloc(&map.y, n * sizeof(float)));
      LaunchBuildMap(MakeParams(options.lens, size.first, size.second,
                                options.zoom),
                     size.first, size.second, map.x, map.y, nullptr);
      device_maps[size] = map;
    }
    UD_CUDA_CHECK(cudaEventRecord(stop));
    UD_CUDA_CHECK(cudaEventSynchronize(stop));
    UD_CUDA_CHECK(cudaEventElapsedTime(&map_ms, start, stop));
    UD_CUDA_CHECK(cudaEventDestroy(start));
    UD_CUDA_CHECK(cudaEventDestroy(stop));
  }

  const int num_slots = std::max(
      1, std::min(options.num_streams, static_cast<int>(images.size())));
  std::vector<StreamSlot> slots(num_slots);
  for (StreamSlot& slot : slots) {
    UD_CUDA_CHECK(
        cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
    UD_CUDA_CHECK(cudaEventCreate(&slot.ev_begin));
    UD_CUDA_CHECK(cudaEventCreate(&slot.ev_kernel));
    UD_CUDA_CHECK(cudaEventCreate(&slot.ev_kdone));
    UD_CUDA_CHECK(cudaEventCreate(&slot.ev_end));
    UD_CUDA_CHECK(cudaMallocHost(&slot.h_in, max_bytes));
    UD_CUDA_CHECK(cudaMallocHost(&slot.h_out, max_bytes));
    UD_CUDA_CHECK(cudaMalloc(&slot.d_in, max_bytes));
    UD_CUDA_CHECK(cudaMalloc(&slot.d_out, max_bytes));
    if (options.backend == Backend::kNpp) {
      slot.npp_ctx = MakeNppContext(slot.stream);
    }
  }
  std::printf("streams            : %d (pinned buffers %.1f MB each)\n",
              num_slots, max_bytes / 1048576.0);
  std::printf("-----------------------------------------\n");

  CpuMapCache cpu_maps(options.lens, options.zoom);
  std::vector<uint8_t> reference;

  auto enqueue = [&](StreamSlot* slot, int index) {
    const ImageInfo& info = images[index];
    const int w = info.width, h = info.height;
    const size_t bytes = static_cast<size_t>(w) * h * 3;
    UD_CUDA_CHECK(cudaEventRecord(slot->ev_begin, slot->stream));
    UD_CUDA_CHECK(cudaMemcpyAsync(slot->d_in, slot->h_in, bytes,
                                  cudaMemcpyHostToDevice, slot->stream));
    UD_CUDA_CHECK(cudaEventRecord(slot->ev_kernel, slot->stream));
    switch (options.backend) {
      case Backend::kKernelMap: {
        const DeviceMap& map = device_maps.at(SizeKey(w, h));
        LaunchRemap(slot->d_in, w, h, map.x, map.y, options.interp,
                    slot->d_out, slot->stream);
        break;
      }
      case Backend::kKernelFused:
        LaunchFusedUndistort(slot->d_in, w, h,
                             MakeParams(options.lens, w, h, options.zoom),
                             options.interp, slot->d_out, slot->stream);
        break;
      case Backend::kNpp: {
        const DeviceMap& map = device_maps.at(SizeKey(w, h));
        NppRemap(slot->d_in, w, h, map.x, map.y, options.interp, slot->d_out,
                 slot->npp_ctx);
        break;
      }
      case Backend::kCpu:
        break;
    }
    UD_CUDA_CHECK(cudaEventRecord(slot->ev_kdone, slot->stream));
    UD_CUDA_CHECK(cudaMemcpyAsync(slot->h_out, slot->d_out, bytes,
                                  cudaMemcpyDeviceToHost, slot->stream));
    UD_CUDA_CHECK(cudaEventRecord(slot->ev_end, slot->stream));
    slot->image_index = index;
  };

  auto finish = [&](StreamSlot* slot) {
    if (slot->image_index < 0) return;
    const int index = slot->image_index;
    const ImageInfo& info = images[index];
    UD_CUDA_CHECK(cudaStreamSynchronize(slot->stream));
    float h2d = 0.0f, kernel = 0.0f, d2h = 0.0f;
    UD_CUDA_CHECK(cudaEventElapsedTime(&h2d, slot->ev_begin, slot->ev_kernel));
    UD_CUDA_CHECK(cudaEventElapsedTime(&kernel, slot->ev_kernel,
                                       slot->ev_kdone));
    UD_CUDA_CHECK(cudaEventElapsedTime(&d2h, slot->ev_kdone, slot->ev_end));

    DiffStats diff;
    const DiffStats* diff_ptr = nullptr;
    if (index < options.verify_count) {
      const size_t bytes = static_cast<size_t>(info.width) * info.height * 3;
      reference.resize(bytes);
      const CpuMap& map = cpu_maps.Get(info.width, info.height);
      const auto start = Clock::now();
      RemapCpu(slot->h_in, info.width, info.height, map.x.data(),
               map.y.data(), options.interp, reference.data());
      sink.AddCpuReferenceTime(MillisSince(start), kernel);
      diff = Compare(slot->h_out, reference.data(), bytes);
      diff_ptr = &diff;
    }
    sink.Add(index, info, slot->h_in, slot->h_out, kernel, h2d + d2h,
             diff_ptr);
    slot->image_index = -1;
  };

  const auto wall_start = Clock::now();
  for (size_t i = 0; i < images.size(); ++i) {
    StreamSlot& slot = slots[i % num_slots];
    finish(&slot);  // frees the slot; other streams keep the GPU busy
    int w = 0, h = 0;
    if (!ReadPpmInto(images[i].path, slot.h_in, max_bytes, &w, &h, &error)) {
      std::printf("  [error] %s\n", error.c_str());
      sink.AddError();
      continue;
    }
    enqueue(&slot, static_cast<int>(i));
  }
  // Drain in submission order so the log stays sorted.
  std::vector<StreamSlot*> pending;
  for (StreamSlot& slot : slots) {
    if (slot.image_index >= 0) pending.push_back(&slot);
  }
  std::sort(pending.begin(), pending.end(),
            [](const StreamSlot* a, const StreamSlot* b) {
              return a->image_index < b->image_index;
            });
  for (StreamSlot* slot : pending) finish(slot);
  const double wall_ms = MillisSince(wall_start);

  const bool ok = sink.PrintSummary(wall_ms, map_ms);

  for (StreamSlot& slot : slots) {
    UD_CUDA_CHECK(cudaFreeHost(slot.h_in));
    UD_CUDA_CHECK(cudaFreeHost(slot.h_out));
    UD_CUDA_CHECK(cudaFree(slot.d_in));
    UD_CUDA_CHECK(cudaFree(slot.d_out));
    UD_CUDA_CHECK(cudaEventDestroy(slot.ev_begin));
    UD_CUDA_CHECK(cudaEventDestroy(slot.ev_kernel));
    UD_CUDA_CHECK(cudaEventDestroy(slot.ev_kdone));
    UD_CUDA_CHECK(cudaEventDestroy(slot.ev_end));
    UD_CUDA_CHECK(cudaStreamDestroy(slot.stream));
  }
  for (auto& kv : device_maps) {
    UD_CUDA_CHECK(cudaFree(kv.second.x));
    UD_CUDA_CHECK(cudaFree(kv.second.y));
  }
  return ok ? 0 : 1;
}

}  // namespace

int RunBatch(const Options& options) {
  std::vector<ImageInfo> images;
  std::string error;
  if (!ListPpmImages(options.input_dir, &images, &error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return 1;
  }
  if (options.limit >= 0 && images.size() > static_cast<size_t>(options.limit))
    images.resize(options.limit);
  if (images.empty()) {
    std::fprintf(stderr, "error: no .ppm images in %s\n",
                 options.input_dir.c_str());
    return 1;
  }
  std::printf("=========== CUDA lens undistort ===========\n");
  std::printf("input              : %s (%zu images)\n",
              options.input_dir.c_str(), images.size());
  std::printf("output             : %s\n",
              options.write_outputs ? options.output_dir.c_str()
                                    : "(not written)");
  std::printf("lens               : %s zoom=%.3f\n",
              LensToString(options.lens).c_str(), options.zoom);
  std::printf("backend / interp   : %s / %s\n", BackendName(options.backend),
              InterpName(options.interp));
  if (options.backend == Backend::kCpu) return RunCpu(options, images);
  return RunGpu(options, images);
}

}  // namespace undistort
