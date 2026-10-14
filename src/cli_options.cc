// Copyright 2026 Divjot

#include "src/cli_options.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace undistort {
namespace {

bool ParseInt(const std::string& text, int* out) {
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);  // NOLINT
  if (end == text.c_str() || *end != '\0') return false;
  *out = static_cast<int>(value);
  return true;
}

bool ParseFloat(const std::string& text, float* out) {
  char* end = nullptr;
  *out = std::strtof(text.c_str(), &end);
  return end != text.c_str() && *end == '\0';
}

bool ParseBackend(const std::string& text, Backend* out) {
  if (text == "kernel_map") {
    *out = Backend::kKernelMap;
  } else if (text == "kernel_fused") {
    *out = Backend::kKernelFused;
  } else if (text == "npp") {
    *out = Backend::kNpp;
  } else if (text == "cpu") {
    *out = Backend::kCpu;
  } else {
    return false;
  }
  return true;
}

}  // namespace

const char* BackendName(Backend backend) {
  switch (backend) {
    case Backend::kKernelMap:
      return "kernel_map";
    case Backend::kKernelFused:
      return "kernel_fused";
    case Backend::kNpp:
      return "npp";
    case Backend::kCpu:
      return "cpu";
  }
  return "unknown";
}

const char* InterpName(Interp interp) {
  return interp == Interp::kBilinear ? "bilinear" : "nearest";
}

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s --input_dir DIR [options]\n"
      "\n"
      "Batch lens-distortion correction of PPM images on the GPU.\n"
      "\n"
      "Required:\n"
      "  --input_dir DIR      directory of distorted *.ppm images\n"
      "\n"
      "Lens model (Brown-Conrady):\n"
      "  --camera FILE        key=value file (focal_scale,k1,k2,k3,p1,p2)\n"
      "  --focal_scale F      fx = fy = F * width            (default 0.6)\n"
      "  --k1 --k2 --k3 F     radial coefficients            (default 0)\n"
      "  --p1 --p2 F          tangential coefficients        (default 0)\n"
      "                       (individual flags override the camera file)\n"
      "  --zoom F             output focal multiplier        (default 1.0)\n"
      "\n"
      "Processing:\n"
      "  --backend NAME       kernel_map | kernel_fused | npp | cpu\n"
      "                                                      (default "
      "kernel_map)\n"
      "  --interp NAME        bilinear | nearest             (default "
      "bilinear)\n"
      "  --streams N          concurrent CUDA streams        (default 4)\n"
      "  --device N           CUDA device index              (default 0)\n"
      "  --limit N            process at most N images       (default all)\n"
      "\n"
      "Output and checks:\n"
      "  --output_dir DIR     where corrected images go "
      "(default output/corrected)\n"
      "  --no_write           skip writing corrected images (benchmarking)\n"
      "  --gt_dir DIR         ground-truth images (same names) -> PSNR\n"
      "  --verify N           compare first N GPU results with CPU reference\n"
      "                       (default 8, 0 disables)\n"
      "  --csv FILE           write per-image metrics as CSV\n"
      "  --quiet              only print the summary\n"
      "  --help               show this message\n",
      program);
}

bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error) {
  *show_help = false;
  std::vector<std::pair<std::string, float>> lens_overrides;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      *error = "unexpected argument: " + arg;
      return false;
    }
    std::string name = arg.substr(2);
    std::string value;
    bool has_value = false;
    const size_t eq = name.find('=');
    if (eq != std::string::npos) {
      value = name.substr(eq + 1);
      name = name.substr(0, eq);
      has_value = true;
    }
    // Boolean flags.
    if (name == "help" || name == "h") {
      *show_help = true;
      return true;
    }
    if (name == "no_write") {
      options->write_outputs = false;
      continue;
    }
    if (name == "quiet") {
      options->quiet = true;
      continue;
    }
    // Everything else takes a value.
    if (!has_value) {
      if (i + 1 >= argc) {
        *error = "missing value for --" + name;
        return false;
      }
      value = argv[++i];
    }
    bool ok = true;
    float fval = 0.0f;
    if (name == "input_dir") {
      options->input_dir = value;
    } else if (name == "output_dir") {
      options->output_dir = value;
    } else if (name == "camera") {
      options->camera_file = value;
    } else if (name == "gt_dir") {
      options->gt_dir = value;
    } else if (name == "csv") {
      options->csv_path = value;
    } else if (name == "backend") {
      ok = ParseBackend(value, &options->backend);
    } else if (name == "interp") {
      if (value == "bilinear") {
        options->interp = Interp::kBilinear;
      } else if (value == "nearest") {
        options->interp = Interp::kNearest;
      } else {
        ok = false;
      }
    } else if (name == "zoom") {
      ok = ParseFloat(value, &options->zoom) && options->zoom > 0.0f;
    } else if (name == "streams") {
      ok = ParseInt(value, &options->num_streams) &&
           options->num_streams >= 1 && options->num_streams <= 32;
    } else if (name == "device") {
      ok = ParseInt(value, &options->device) && options->device >= 0;
    } else if (name == "limit") {
      ok = ParseInt(value, &options->limit);
    } else if (name == "verify") {
      ok = ParseInt(value, &options->verify_count) &&
           options->verify_count >= 0;
    } else if (name == "focal_scale" || name == "k1" || name == "k2" ||
               name == "k3" || name == "p1" || name == "p2") {
      ok = ParseFloat(value, &fval);
      if (ok) lens_overrides.emplace_back(name, fval);
    } else {
      *error = "unknown flag --" + name;
      return false;
    }
    if (!ok) {
      *error = "invalid value for --" + name + ": '" + value + "'";
      return false;
    }
  }
  if (options->input_dir.empty()) {
    *error = "--input_dir is required";
    return false;
  }
  if (!options->camera_file.empty() &&
      !LoadLensConfig(options->camera_file, &options->lens, error)) {
    return false;
  }
  for (const auto& kv : lens_overrides) {
    SetLensField(kv.first, kv.second, &options->lens);
  }
  if (options->lens.focal_scale <= 0.0f) {
    *error = "focal_scale must be positive";
    return false;
  }
  return true;
}

}  // namespace undistort
