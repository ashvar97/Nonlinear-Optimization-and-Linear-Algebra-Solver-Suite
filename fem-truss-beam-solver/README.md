# FEM Truss & Beam Solver

MATLAB/Octave scripts implementing the Finite Element Method (FEM) for two classic 1D structural
problems: an axially-loaded truss/rod ("Stab") clamped at both ends, and a cantilever (Bernoulli)
beam. Originally coursework for a Finite Elements lecture; verified here against closed-form
analytical solutions before being folded into this repo.

Both solvers follow the same textbook FEM pattern used by `optimization-suite/`'s linear-algebra
routines: discretize -> assemble a global stiffness matrix `K` and load vector `F` element by
element -> solve the resulting linear system `K*d = F` under Dirichlet boundary conditions. They
share one `solveq.m` for that last step (originally two copy-pasted, functionally-identical
copies — `solveq.m` and `SP_solveq.m` — consolidated into one, since eliminating Dirichlet DOFs
from a linear system doesn't depend on which element type produced `K`/`F`).

## Two solvers

### 1. Truss/rod under a linearly varying distributed load (`SP_*.m`)

A rod of length `l`, fixed at both ends, under a linearly distributed axial load `n(x) = k*x`.
Supports **linear** (2-node) or **quadratic** (3-node) 1D elements.

- `SP_stab_diskret.m` — mesh generation: node coordinates, connectivity, Dirichlet BCs.
- `SP_elem_1d.m` — element stiffness matrix and load vector (linear or quadratic).
- `solveq.m` — solves `K*d = F` with the fixed-end Dirichlet BCs eliminated (shared, see above).
- `SP_main_stab_1d.m` — driver script: prompts for mesh size / element order / load, assembles,
  solves, plots the numerical solution and stresses against the analytical solution.

**Verified**: assembling and solving with 20 elements, both `pol=1` (linear) and `pol=2`
(quadratic), reproduces the closed-form midpoint displacement
`u(l/2) = k*l/(6*E*A) * (l^2 - (l/2)^2)` to within floating-point precision (relative error
`< 1e-13`).

### 2. Cantilever beam under combined loading (`balken_*`, `elem_balken.m`, `solveq.m`, `main_balken.m`)

A cantilever beam under a combination of a uniform line load `q`, a tip point load `Q`, and a tip
moment `M`, using 2-node Bernoulli beam elements (2 DOF/node: deflection + rotation).

- `balken_diskret.m` — mesh generation for the beam.
- `elem_balken.m` — Bernoulli beam element stiffness matrix and consistent load vector.
- `solveq.m` — solves `K*d = F` with the clamped-end Dirichlet BCs eliminated (shared, see above).
- `main_balken.m` — driver script: prompts for element count and loading, assembles, solves,
  plots the numerical deflection against the analytical beam solution.

**Verified**: for a pure tip point load, the numerical tip deflection matches the analytical
cantilever formula `Q*L^3/(3*E*I)` to within `1e-14` relative error.

## Requirements

MATLAB, or [GNU Octave](https://octave.org/) (free, MATLAB-compatible — used to verify these
scripts; no MATLAB-only syntax is used).

## Usage

Both drivers are interactive scripts (they prompt for parameters via `input()`), so run them
from the MATLAB/Octave prompt rather than piping input:

```matlab
cd src
main_balken       % prompts for: number of elements, line load q, point load Q, moment M, EI
SP_main_stab_1d   % prompts for: number of elements, element order (1 or 2), load factor k
```

Each prints/plots the numerical solution against the analytical one for direct comparison.

To drive either solver non-interactively (e.g. for scripting or regression checks), call the
underlying functions directly instead of the driver script — see the verification snippets
above for the exact call sequence (`*_diskret` -> loop assembling `*_elem_*`/`elem_balken` into
`K`/`F` -> `solveq`).

## License

[MIT](../LICENSE)

---

*Coursework-derived, verified against analytical solutions; not validated for professional
structural engineering use.*
