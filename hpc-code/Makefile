# HPC / parallel-computing coursework: benchmarks and mini-programs from a
# programming-for-supercomputers course. Each target below builds one standalone binary
# into bin/. `make` (no target) builds everything that doesn't need nvcc; `make cuda` builds
# the CUDA target separately since nvcc isn't always available.
#
# Usage:
#   make            # build every CPU target
#   make cuda       # build the CUDA STREAM benchmark (needs nvcc)
#   make run-vector_triad   # build (if needed) and run a specific target
#   make clean

CC       := gcc
CXX      := g++
NVCC     := nvcc
CFLAGS   := -std=c11 -O2 -Wall -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Iinclude -fopenmp
LDFLAGS  := -lm
NVCCFLAGS := -O3 -arch=sm_80

BIN_DIR := bin
SRC_DIR := src

# C targets (each needs timing.c linked in for getTimeStamp())
C_TARGETS := vector_triad vector_update stream_triad
# C++/OpenMP targets (self-contained; some link timing.c too, see per-target rules below)
CXX_TARGETS := monte_carlo triangular_mvm thread_scaling_benchmark core_scaling_benchmark \
               barrier_benchmark ray_tracer

.PHONY: all cuda clean run-%
all: $(addprefix $(BIN_DIR)/,$(C_TARGETS) $(CXX_TARGETS))

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/vector_triad: $(SRC_DIR)/vector_triad.c $(SRC_DIR)/timing.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/vector_update: $(SRC_DIR)/vector_update.c $(SRC_DIR)/timing.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/stream_triad: $(SRC_DIR)/stream_triad.c $(SRC_DIR)/timing.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/ray_tracer: $(SRC_DIR)/ray_tracer.cpp $(SRC_DIR)/timing.c | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# monte_carlo, triangular_mvm, thread_scaling_benchmark, core_scaling_benchmark and
# barrier_benchmark define their own getTimeStamp()/use omp_get_wtime() and don't need timing.c.
$(BIN_DIR)/%: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

cuda: $(BIN_DIR)/stream
$(BIN_DIR)/stream: $(SRC_DIR)/stream.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-%: $(BIN_DIR)/%
	./$<

clean:
	rm -rf $(BIN_DIR)
