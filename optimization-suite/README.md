# optimization-suite

A small, dependency-light nonlinear optimization and linear algebra library: Newton, BFGS,
Levenberg-Marquardt and inexact Newton-CG descent methods, conjugate-gradient linear solvers,
box-constrained and augmented-Lagrangian methods for constrained problems, plus the line
searches, Hessian approximations and test objectives they're built on.

Originally written as course material for "Optimization for Engineers" (Dr. Johannes Hild);
every algorithm here ships with a worked, hand-checkable example (see [Testing](#testing)).

## What's fixed here

The suite as originally written had two modules referenced everywhere but implemented nowhere:
`projectionInBox` (a box-projection class) and `projectedHessApprox` (a constrained Hessian
approximation). Without them, three algorithms -- `projectedBFGSDescent`,
`projectedInexactNewtonCG`, and `augmentedLagrangianDescent` -- raised `ModuleNotFoundError` on
import and could not run at all. Both are now implemented (spec inferred from the existing
docstrings/test cases across the codebase) and verified against every documented test case; see
`tests/test_descent.py` for the regression coverage.

Along the way, a couple of latent robustness bugs turned up and were fixed: `BFGSDescent`,
`projectedBFGSDescent` and `implicitFiltering`'s inner BFGS update had no curvature-condition
guard, so a degenerate step (`y.T @ s == 0`) crashed with a raw `ZeroDivisionError` instead of
degrading gracefully (now: skip that update, keep the previous Hessian approximation, matching
standard BFGS practice). `projectedBacktrackingSearch` could in principle backtrack forever if no
step satisfies its condition; it now gives up after a bounded number of halvings.
`projectedBFGSDescent`'s outer loop had no iteration cap (every other iterative method here has
one); it now stops after 500 iterations rather than potentially running forever on a
badly-conditioned problem.

## Package structure

```
optimization-suite/
├── optimization/               # the installable package
│   ├── __init__.py             # re-exports every public name below
│   │
│   ├── incompleteCholesky.py   # sparsity-preserving Cholesky preconditioner
│   ├── LLTSolver.py            # forward/backward substitution for L L^T y = r
│   ├── CGSolver.py             # conjugate gradient
│   ├── PrecCGSolver.py         # preconditioned CG (uses the two above)
│   │
│   ├── bananaValleyObjective.py       # Rosenbrock-style 2D test function
│   ├── quadraticObjective.py          # n-dim quadratic 0.5 x'Ax + b'x + c
│   ├── simpleValleyObjective.py       # 2D test function with a parameter vector
│   ├── modelObjective.py              # 3D nonlinear test function
│   ├── noHessianObjective.py          # 2D function with a local max + 2 minima, no analytic Hessian
│   ├── leastSquaresObjective.py       # residual()/jacobian() for curve fitting
│   ├── leastSquaresModel.py           # (equivalent least-squares wrapper)
│   ├── augmentedLagrangianObjective.py
│   │
│   ├── projectionInBox.py             # box constraint: .project() / .activeIndexSet()
│   │
│   ├── WolfePowellSearch.py           # unconstrained line search
│   ├── projectedBacktrackingSearch.py # box-constrained line search
│   ├── directionalHessApprox.py       # central-difference Hessian-vector product
│   ├── projectedHessApprox.py         # ... restricted to the free (inactive) coordinates
│   ├── SUCSimplexGradient.py          # derivative-free gradient + stencil-failure check
│   │
│   ├── NewtonDescent.py
│   ├── inexactNewtonCG.py
│   ├── projectedInexactNewtonCG.py
│   ├── BFGSDescent.py
│   ├── projectedBFGSDescent.py
│   ├── levenbergMarquardtDescent.py   # nonlinear least squares
│   ├── implicitFiltering.py           # derivative-free BFGS, multi-scale
│   └── augmentedLagrangianDescent.py  # equality-constrained + box-constrained
│
├── tests/                       # pytest suite, one file per category above
├── examples/demo.py             # Newton vs. BFGS + a box-constrained example, with plots
├── legacy/                      # earlier draft variants, kept for reference (see its README)
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # the library itself (just numpy)
pip install -r requirements-dev.txt   # + pytest, flake8, matplotlib (for examples/)
```

## Quick usage

```python
import numpy as np
from optimization import BFGSDescent, bananaValleyObjective

x0 = np.array([[0], [1]], dtype=float)
xmin = BFGSDescent(bananaValleyObjective(), x0, eps=1.0e-6)
print(xmin)  # close to [[1], [1]]
```

Box-constrained descent (the previously-broken path):

```python
import numpy as np
from optimization import projectionInBox, projectedBFGSDescent, simpleValleyObjective

p = np.array([[1], [1]])
objective = simpleValleyObjective(p)
box = projectionInBox(a=np.array([[1], [1]]), b=np.array([[2], [2]]))
x0 = np.array([[2], [2]], dtype=float)

xmin = projectedBFGSDescent(objective, box, x0, eps=1.0e-3)
print(xmin)  # close to [[1], [1]]
```

Every function accepts an `objective` (or `model`, for least-squares) argument that is any
object exposing the methods it needs (`.objective()`, `.gradient()`, sometimes `.hessian()`) --
write your own by matching the interface of e.g. `quadraticObjective` or `bananaValleyObjective`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

46 tests, each one lifted directly from a worked example in the corresponding module's
docstring (input, expected output, tolerance) -- so the test suite doubles as executable
documentation of what every function is supposed to do.

## Examples

```bash
pip install -r requirements-dev.txt   # needs matplotlib
python examples/demo.py
```

Produces `examples/banana_valley_descent.png`: Newton's method converging in a handful of crisp
steps vs. BFGS's slower, wandering path on the same problem, plus a box-constrained example
showing `projectedBFGSDescent` bending its path to hug a feasibility boundary rather than
reaching an infeasible unconstrained optimum.

## License

See the repository root [LICENSE](../LICENSE).
