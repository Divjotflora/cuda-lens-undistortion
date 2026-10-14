// Copyright 2026 Divjot
//
// Minimal binary PPM (P6, 8-bit RGB) I/O. PPM keeps the project free of
// image-library dependencies, and reading straight into a caller-supplied
// buffer lets the pipeline decode directly into pinned host memory.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace undistort {

struct ImageInfo {
  std::string path;
  std::string name;  // file name without directory
  int width = 0;
  int height = 0;
};

// Returns the sorted list of *.ppm files in |dir| with their dimensions.
bool ListPpmImages(const std::string& dir, std::vector<ImageInfo>* images,
                   std::string* error);

bool ReadPpmHeader(const std::string& path, int* width, int* height,
                   std::string* error);

// Reads pixel data into |buffer| (capacity bytes). Rows are tightly packed.
bool ReadPpmInto(const std::string& path, uint8_t* buffer, size_t capacity,
                 int* width, int* height, std::string* error);

bool ReadPpm(const std::string& path, std::vector<uint8_t>* pixels, int* width,
             int* height, std::string* error);

bool WritePpm(const std::string& path, const uint8_t* pixels, int width,
              int height, std::string* error);

bool MakeDirs(const std::string& dir, std::string* error);

}  // namespace undistort

#endif  // SRC_IMAGE_IO_H_
