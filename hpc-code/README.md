# hpc-code

Parallel-computing coursework: a set of standalone C/C++/OpenMP/CUDA micro-benchmarks and small
programs from a "Programming for Supercomputers" course. Each `.c`/`.cpp` file is an independent
`main()` -- there's no single unifying program, just related benchmarking exercises, so each
builds to its own binary.

## What's fixed here

Three real bugs turned up while getting everything to actually build and run:

- **`core_scaling_benchmark.cpp` segfaulted immediately.** It declared `double a[N], b[N]` with
  `N = 100000000` as plain stack arrays -- ~1.6GB on the stack, which overflows on any normal
  stack-size limit before the program does anything. Fixed to heap-allocate (`new`/`delete[]`),
  matching how the other benchmarks here already do it.
- **`ray_tracer.cpp` used OpenMP (`omp_set_num_threads`, `#pragma omp parallel`) without
  including `<omp.h>`**, so it only compiled by accident if some other header transitively pulled
  it in. Added the include.
- **`barrier_benchmark.cpp`** (originally `Barrier.cpp`) was truncated: minified to a single line
  with no closing braces for its outer loop or `main()`, so it didn't compile at all. Recovered
  and reformatted; also dropped a benign but pointless data race (an unused `sum` accumulated
  from inside a `#pragma omp parallel for` with no reduction clause).

Every target below now builds cleanly with `-Wall` (warnings-only, no errors) and was smoke-run
to confirm it doesn't crash.

## Layout

```
hpc-code/
├── src/
│   ├── vector_triad.c              # a[i] = b[i] + c[i]*d[i], swept over 37 vector lengths
│   ├── vector_update.c             # a[i] = s*a[i],           swept over 37 vector lengths
│   ├── stream_triad.c              # STREAM-style triad, swept over stride M
│   ├── monte_carlo.cpp             # OpenMP Monte Carlo estimate of pi (points-in-circle)
│   ├── triangular_mvm.cpp          # OpenMP parallel triangular matrix-vector multiply
│   ├── thread_scaling_benchmark.cpp  # Gflop/s of a vector update vs. OpenMP thread count
│   ├── core_scaling_benchmark.cpp    # same idea, swept up to 16 threads, averaged over 10 runs
│   ├── barrier_benchmark.cpp         # OpenMP parallel-for overhead vs. thread count (1..72)
│   ├── ray_tracer.cpp              # small OpenMP-parallel ray tracer, writes a .pgm image
│   ├── stream.cu                   # CUDA STREAM-triad-style bandwidth benchmark
│   └── timing.c                    # shared getTimeStamp()/getTimeResolution() helpers
├── include/timing.h
├── results/                        # captured output from previous benchmark runs
│   ├── v_triad.txt
│   ├── triangular_output.txt
│   ├── performance_vs_cores.txt
│   └── cores.txt
├── bench.sh                        # sweeps the CUDA stream binary over a range of block counts
└── Makefile
```

## Building and running

```bash
make            # builds every CPU (C/C++/OpenMP) target into bin/
make cuda       # builds bin/stream (needs nvcc + an NVIDIA GPU; not built by `make all`)
make run-vector_triad     # build (if needed) and run a specific target
make clean
```

Requires `gcc`/`g++` with OpenMP support (`-fopenmp`, standard on Linux) and, for the CUDA
target only, the CUDA toolkit.

Several of these (`vector_triad`, `vector_update`, `stream_triad`, `monte_carlo`,
`core_scaling_benchmark`) are intentionally long-running -- they sweep many sizes/thread counts
to produce a performance curve, not a single quick answer. `results/` holds example output from
earlier runs if you just want to see representative numbers without waiting.

## License

See the repository root [LICENSE](../LICENSE).
