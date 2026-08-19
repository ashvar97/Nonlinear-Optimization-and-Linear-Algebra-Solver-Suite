# Nonlinear FEM

A geometrically-linear but *materially nonlinear* 1D truss, solved by minimizing total potential
energy with `optimization-suite`'s own `NewtonDescent` and `BFGSDescent` -- the one connection
this repository's name promises (**Nonlinear Optimization** *and* a **Linear Algebra / FEM
Solver Suite**, working *together*) that nothing else here actually delivers.
`fem-truss-beam-solver/` only ever assembles `K*d = F` and solves it directly, because every
element there is linear-elastic; the moment an element's force-displacement law stops being a
straight line, a direct linear solve no longer applies, and you need exactly what's here.

## The physics

A chain of `N` bar elements in series, fixed at one end, with a point force `F` at the free end.
Instead of Hooke's law, each element has a **cubic-hardening** force law:

```
force(e) = k*e + k3*e**3        (e = elongation of the element)
```

with strain-energy density chosen so its derivative gives exactly that force:

```
psi(e) = 0.5*k*e**2 + 0.25*k3*e**4
```

Total potential energy of the chain, as a function of the `N` free-DOF displacements:

```
Pi(d) = sum_i psi(e_i) - F*d_N,   e_i = d_i - d_{i-1}
```

By the **principle of minimum potential energy**, the equilibrium configuration is the `d` that
*minimizes* `Pi` -- exactly the problem `optimization-suite`'s descent methods are built to
solve. `NonlinearTrussObjective` in `nonlinear_truss.py` implements `Pi`, its gradient, and its
(exact, analytic) Hessian in the `.objective()/.gradient()/.hessian()` shape every
`optimization-suite` descent method expects. Setting `k3=0` collapses this back to an ordinary
linear FEM truss, which is exactly how it's checked for correctness (see below).

## Verification

`tests/test_nonlinear_truss.py` (8 tests, all passing) checks the solver against something
computed a *different* way each time, not against itself:

- **Single element** against the closed-form real root of the cubic `k*e + k3*e^3 = F`.
- **Linear limit** (`k3=0`, `N` elements) against the textbook series-spring formula
  `d_N = F*N/k`.
- **`NewtonDescent` vs. `BFGSDescent`** agree with each other to `< 1e-6`.
- Against an **independent, from-scratch Newton-Raphson root-finder** that shares no code with
  `optimization-suite` (agrees to `< 1e-9`).
- Gradient norm at the solution is `< 1e-8` (first-order optimality).
- A **hardening sanity check**: the same tip force produces *less* displacement in the nonlinear
  truss than in its linear part alone -- otherwise it wouldn't be "hardening."
- Zero force gives zero displacement.
- The analytic Hessian matches a central finite-difference approximation to `< 1e-3`.

## A real bug found and fixed along the way

Building this surfaced a genuine, previously-latent bug in `optimization-suite`'s
`WolfePowellSearch.py`: its bisection loop had no stall/iteration guard. Near a converged point
where the gradient is already tiny, the search direction's descent measure can be nothing but
floating-point noise, so neither Wolfe-Powell condition can ever be satisfied *exactly* -- but
the bisection interval still collapses to a single representable `float64` value, and the loop
spun forever. Fixed by detecting when the next candidate step no longer differs from the current
one (bisection has hit floating-point precision) and accepting that step instead of continuing
to loop. Verified this doesn't change any existing behavior: all 46 of `optimization-suite`'s
pre-existing tests still pass unchanged.

## Usage

```bash
python examples/demo.py
```

Sweeps the tip force from just above 0 to 100, plots the nonlinear vs. linear force-displacement
curve (the hardening effect: at a given force, the nonlinear truss displaces less), and a second
panel comparing how many gradient evaluations `NewtonDescent` vs. `BFGSDescent` need to reach
the same equilibrium at each force level -- Newton, with its exact analytic Hessian, converges
in a handful of iterations regardless of force; BFGS's gradient-only quasi-Newton updates need
tens to hundreds.

```python
from nonlinear_truss import solve

d = solve(N=6, k=80.0, k3=15.0, F=40.0, method="newton")
```

## Requirements

```bash
pip install numpy matplotlib pytest
```

## License

[MIT](../LICENSE)
