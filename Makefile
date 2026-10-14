# CUDA batch lens undistortion
#
#   make            build bin/lens_undistort
#   make data       generate the synthetic dataset (needs python3 + numpy)
#   make run        build, generate data if needed and run the full demo
#   make lint       check Google C++ style with cpplint
#   make clean      remove build products
#
# Override variables on the command line, e.g.
#   make ARCH="-arch=sm_75"          build for one GPU only
#   make CCBIN=g++-12                pick the host compiler used by nvcc

NVCC      ?= nvcc
CCBIN     ?=
ARCH      ?= -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_60,code=sm_60 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=[sm_86,compute_86]

BIN_DIR   := bin
BUILD_DIR := build
TARGET    := $(BIN_DIR)/lens_undistort

NVCCFLAGS := -std=c++17 -O3 -lineinfo -I. $(ARCH) \
             -Xcompiler -Wall,-Wextra
ifneq ($(CCBIN),)
NVCCFLAGS += -ccbin $(CCBIN)
endif
LDLIBS    := -lnppig -lnppc

CU_SRCS   := src/main.cu src/batch_pipeline.cu src/undistort_gpu.cu
CC_SRCS   := src/camera_model.cc src/cli_options.cc src/image_io.cc \
             src/metrics.cc src/undistort_cpu.cc
OBJS      := $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o) \
             $(CC_SRCS:src/%.cc=$(BUILD_DIR)/%.o)

.PHONY: all clean data run lint help

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cu src/*.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cc src/*.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

data:
	python3 scripts/make_dataset.py --out data --count 200 \
	    --width 800 --height 600

run: $(TARGET)
	./run.sh

lint:
	cpplint --extensions=cc,h,cu --recursive src

help:
	@sed -n '1,14p' Makefile

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
