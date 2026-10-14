// Copyright 2026 Divjot

#include "src/image_io.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace undistort {
namespace {

// Reads the next whitespace-delimited token, skipping '#' comments.
bool NextToken(std::istream* in, std::string* token) {
  token->clear();
  int c;
  while ((c = in->get()) != EOF) {
    if (c == '#') {
      while ((c = in->get()) != EOF && c != '\n') {
      }
      continue;
    }
    if (!std::isspace(c)) break;
  }
  if (c == EOF) return false;
  do {
    token->push_back(static_cast<char>(c));
  } while ((c = in->get()) != EOF && !std::isspace(c));
  // The single whitespace byte after maxval has now been consumed, which is
  // exactly what the PPM format requires before the binary raster.
  return true;
}

bool ParseHeader(std::istream* in, const std::string& path, int* width,
                 int* height, std::string* error) {
  std::string magic, w, h, maxval;
  if (!NextToken(in, &magic) || magic != "P6") {
    *error = path + ": not a binary PPM (P6)";
    return false;
  }
  if (!NextToken(in, &w) || !NextToken(in, &h) || !NextToken(in, &maxval)) {
    *error = path + ": truncated PPM header";
    return false;
  }
  *width = std::atoi(w.c_str());
  *height = std::atoi(h.c_str());
  if (*width <= 0 || *height <= 0 || maxval != "255") {
    *error = path + ": unsupported PPM (need 8-bit, positive size)";
    return false;
  }
  return true;
}

}  // namespace

bool ReadPpmHeader(const std::string& path, int* width, int* height,
                   std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  return ParseHeader(&in, path, width, height, error);
}

bool ReadPpmInto(const std::string& path, uint8_t* buffer, size_t capacity,
                 int* width, int* height, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  if (!ParseHeader(&in, path, width, height, error)) return false;
  const size_t bytes = static_cast<size_t>(*width) * *height * 3;
  if (bytes > capacity) {
    *error = path + ": image larger than buffer";
    return false;
  }
  in.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(in.gcount()) != bytes) {
    *error = path + ": truncated pixel data";
    return false;
  }
  return true;
}

bool ReadPpm(const std::string& path, std::vector<uint8_t>* pixels, int* width,
             int* height, std::string* error) {
  if (!ReadPpmHeader(path, width, height, error)) return false;
  pixels->resize(static_cast<size_t>(*width) * *height * 3);
  return ReadPpmInto(path, pixels->data(), pixels->size(), width, height,
                     error);
}

bool WritePpm(const std::string& path, const uint8_t* pixels, int width,
              int height, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot write " + path;
    return false;
  }
  out << "P6\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(pixels),
            static_cast<std::streamsize>(width) * height * 3);
  if (!out) {
    *error = "write failed: " + path;
    return false;
  }
  return true;
}

bool ListPpmImages(const std::string& dir, std::vector<ImageInfo>* images,
                   std::string* error) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    *error = "input directory does not exist: " + dir;
    return false;
  }
  images->clear();
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".ppm") continue;
    ImageInfo info;
    info.path = entry.path().string();
    info.name = entry.path().filename().string();
    if (!ReadPpmHeader(info.path, &info.width, &info.height, error)) {
      return false;
    }
    images->push_back(info);
  }
  if (ec) {
    *error = "cannot list " + dir + ": " + ec.message();
    return false;
  }
  std::sort(images->begin(), images->end(),
            [](const ImageInfo& a, const ImageInfo& b) {
              return a.name < b.name;
            });
  return true;
}

bool MakeDirs(const std::string& dir, std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    *error = "cannot create " + dir + ": " + ec.message();
    return false;
  }
  return true;
}

}  // namespace undistort
