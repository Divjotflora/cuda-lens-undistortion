// Copyright 2026 Divjot
//
// Command-line parsing. Flags accept both "--name value" and "--name=value".

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

#include "src/camera_model.h"

namespace undistort {

enum class Backend { kKernelMap, kKernelFused, kNpp, kCpu };

struct Options {
  std::string input_dir;
  std::string output_dir = "output/corrected";
  std::string camera_file;
  std::string gt_dir;    // optional ground truth for PSNR
  std::string csv_path;  // optional per-image metrics
  Backend backend = Backend::kKernelMap;
  Interp interp = Interp::kBilinear;
  LensConfig lens;
  float zoom = 1.0f;
  int num_streams = 4;
  int limit = -1;         // process at most N images (-1 = all)
  int verify_count = 8;   // GPU-vs-CPU checks on the first N images
  int device = 0;
  bool write_outputs = true;
  bool quiet = false;     // suppress per-image lines
};

// Returns false and fills |error| on invalid input. Sets |*show_help| when
// --help was requested.
bool ParseOptions(int argc, char** argv, Options* options, bool* show_help,
                  std::string* error);

void PrintUsage(const char* program);

const char* BackendName(Backend backend);
const char* InterpName(Interp interp);

}  // namespace undistort

#endif  // SRC_CLI_OPTIONS_H_
