// Copyright 2026 Divjot

#include "src/camera_model.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace undistort {

UndistortParams MakeParams(const LensConfig& lens, int width, int height,
                           float zoom) {
  UndistortParams p;
  p.in_fx = lens.focal_scale * width;
  p.in_fy = p.in_fx;
  p.in_cx = 0.5f * (width - 1);
  p.in_cy = 0.5f * (height - 1);
  p.out_fx = p.in_fx * zoom;
  p.out_fy = p.in_fy * zoom;
  p.out_cx = p.in_cx;
  p.out_cy = p.in_cy;
  p.k1 = lens.k1;
  p.k2 = lens.k2;
  p.k3 = lens.k3;
  p.p1 = lens.p1;
  p.p2 = lens.p2;
  return p;
}

bool SetLensField(const std::string& key, float value, LensConfig* lens) {
  if (key == "focal_scale") {
    lens->focal_scale = value;
  } else if (key == "k1") {
    lens->k1 = value;
  } else if (key == "k2") {
    lens->k2 = value;
  } else if (key == "k3") {
    lens->k3 = value;
  } else if (key == "p1") {
    lens->p1 = value;
  } else if (key == "p2") {
    lens->p2 = value;
  } else {
    return false;
  }
  return true;
}

bool LoadLensConfig(const std::string& path, LensConfig* lens,
                    std::string* error) {
  std::ifstream in(path);
  if (!in) {
    *error = "cannot open camera file: " + path;
    return false;
  }
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    std::string val = line.substr(eq + 1);
    std::istringstream key_stream(key);
    key_stream >> key;
    char* end = nullptr;
    const float value = std::strtof(val.c_str(), &end);
    if (end == val.c_str() || !SetLensField(key, value, lens)) {
      *error = path + ":" + std::to_string(line_no) + ": bad entry '" + line +
               "'";
      return false;
    }
  }
  return true;
}

std::string LensToString(const LensConfig& lens) {
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "focal_scale=%.4f k1=%.5f k2=%.5f k3=%.5f p1=%.5f p2=%.5f",
                lens.focal_scale, lens.k1, lens.k2, lens.k3, lens.p1, lens.p2);
  return buf;
}

}  // namespace undistort
