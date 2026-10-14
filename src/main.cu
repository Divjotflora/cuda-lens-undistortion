// Copyright 2026 Divjot
//
// Entry point: parse flags, then hand off to the batch pipeline.

#include <cstdio>
#include <string>

#include "src/batch_pipeline.h"
#include "src/cli_options.h"

int main(int argc, char** argv) {
  undistort::Options options;
  bool show_help = false;
  std::string error;
  if (!undistort::ParseOptions(argc, argv, &options, &show_help, &error)) {
    std::fprintf(stderr, "error: %s\n\n", error.c_str());
    undistort::PrintUsage(argv[0]);
    return 2;
  }
  if (show_help) {
    undistort::PrintUsage(argv[0]);
    return 0;
  }
  return undistort::RunBatch(options);
}
