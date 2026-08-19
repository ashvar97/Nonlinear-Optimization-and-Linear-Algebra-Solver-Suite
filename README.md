# Nonlinear Optimization and Linear Algebra Solver Suite

Four independent university-coursework projects, kept in one repository, all ultimately about
turning a mathematical model into a linear or nonlinear system and solving it numerically:

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

## [`fem-truss-beam-solver/`](fem-truss-beam-solver/)

MATLAB/Octave Finite Element Method solvers for two classic 1D structural problems: a truss/rod
under a linearly varying distributed load (linear or quadratic elements), and a cantilever beam
under combined line load, point load, and moment. Same discretize -> assemble `K`, `F` -> solve
`K*d = F` pattern as `optimization-suite/`'s linear solvers, applied to a physical problem.
Both solvers were verified against their closed-form analytical solutions (relative error
`< 1e-13`). See [`fem-truss-beam-solver/README.md`](fem-truss-beam-solver/README.md).

## [`density-estimation/`](density-estimation/)

Python scripts for nonparametric density estimation (k-NN and Parzen/kernel windows),
likelihood-based hyperparameter selection, and two ensembling strategies (IID resampling and
bootstrap aggregating). Hyperparameter selection here -- choosing `k` or the bandwidth `h` by
maximizing held-out log-likelihood -- is itself a small optimization problem, in the same spirit
as `optimization-suite/`. Originally 8 scripts that were ~90% copy-pasted boilerplate (a KNN and
a Parzen variant of each stage); consolidated into a shared module plus one script per stage, all
run end-to-end and verified error-free. One real compatibility bug was found and fixed along the
way (`np.Inf`, removed in NumPy 2.0, updated to `np.inf`). See
[`density-estimation/README.md`](density-estimation/README.md).

## [`hpc-code/`](hpc-code/)

A separate, unrelated set of parallel-computing coursework: OpenMP and CUDA C/C++
micro-benchmarks (vector triad, Monte Carlo pi estimation, a small ray tracer, thread/core
scaling curves) and a couple of real bugs fixed along the way (a stack-overflow segfault, a
missing header, a truncated source file). See [`hpc-code/README.md`](hpc-code/README.md).

## License

[MIT](LICENSE)
