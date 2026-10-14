// Copyright 2026 Divjot
//
// Streams a directory of images through the selected back end.
//
// GPU design: K "slots", each owning a CUDA stream, pinned host buffers,
// device buffers and timing events. Image i is assigned to slot i % K. While
// the GPU copies and processes the frames in flight on the other streams,
// the CPU finishes the oldest slot (metrics, verification, writing to disk)
// and decodes the next file straight into that slot's pinned buffer. This
// overlaps disk I/O, PCIe transfers and kernel execution.

#ifndef SRC_BATCH_PIPELINE_H_
#define SRC_BATCH_PIPELINE_H_

#include "src/cli_options.h"

namespace undistort {

// Runs the whole batch and prints a log plus a summary to stdout.
// Returns the process exit code (non-zero on I/O or verification failure).
int RunBatch(const Options& options);

}  // namespace undistort

#endif  // SRC_BATCH_PIPELINE_H_
