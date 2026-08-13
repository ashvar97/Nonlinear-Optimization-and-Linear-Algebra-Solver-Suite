# Nonlinear Optimization and Linear Algebra Solver Suite

Two independent university-coursework projects, kept in one repository:

## [`optimization-suite/`](optimization-suite/) -- start here

A nonlinear optimization and linear algebra library in Python: Newton, BFGS,
Levenberg-Marquardt and inexact Newton-CG descent, conjugate-gradient linear solvers, and
box-constrained / augmented-Lagrangian methods for constrained problems -- 26 modules, a 46-test
pytest suite (one test per documented worked example), and a demo script with plots.

This is the part that matches the repository's name, and where the interesting fixes happened:
two modules the rest of the codebase depended on (`projectionInBox`, `projectedHessApprox`) were
referenced everywhere but implemented nowhere, so three of the descent methods couldn't run at
all before now. See [`optimization-suite/README.md`](optimization-suite/README.md) for the full
writeup, algorithm catalog, and usage examples.

## [`hpc-code/`](hpc-code/)

A separate, unrelated set of parallel-computing coursework: OpenMP and CUDA C/C++
micro-benchmarks (vector triad, Monte Carlo pi estimation, a small ray tracer, thread/core
scaling curves) and a couple of real bugs fixed along the way (a stack-overflow segfault, a
missing header, a truncated source file). See [`hpc-code/README.md`](hpc-code/README.md).

## License

[MIT](LICENSE)
