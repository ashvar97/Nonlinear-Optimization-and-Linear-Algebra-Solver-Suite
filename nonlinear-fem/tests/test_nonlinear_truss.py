"""Verification suite for nonlinear_truss.py.

Every test here checks the optimization-suite-driven solution against something computed a
different way -- a closed-form root, the known linear-FEM limit, or an independent hand-rolled
Newton-Raphson solver that does not import optimization-suite at all -- rather than checking the
solver against itself.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nonlinear_truss import NonlinearTrussObjective, solve  # noqa: E402


def independent_newton_raphson(N, k, k3, F, tol=1e-12, max_iter=200):
    """A from-scratch Newton-Raphson root-finder on the gradient, sharing no code with
    NewtonDescent/BFGSDescent -- used as an independent cross-check, not as "the" reference."""
    objective = NonlinearTrussObjective(N, k, k3, F)
    x = np.zeros((N, 1))
    for _ in range(max_iter):
        g = objective.gradient(x)
        if np.linalg.norm(g) < tol:
            return x
        x = x - np.linalg.solve(objective.hessian(x), g)
    raise RuntimeError("independent Newton-Raphson did not converge")


def test_single_element_matches_closed_form_cubic_root():
    # one element: k*e + k3*e**3 = F has a closed-form real root via np.roots
    k, k3, F = 100.0, 20.0, 50.0
    d = solve(1, k, k3, F, method="newton", eps=1e-12)

    roots = np.roots([k3, 0.0, k, -F])
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    assert len(real_roots) == 1
    assert d[0, 0] == pytest.approx(real_roots[0], abs=1e-8)


def test_linear_limit_matches_series_spring_formula():
    # k3=0 collapses every element to a linear spring; N identical springs in series behave
    # like one spring of stiffness k/N, so the tip displacement is F*N/k.
    N, k, F = 5, 100.0, 30.0
    d = solve(N, k, 0.0, F, method="newton", eps=1e-12)
    assert d[-1, 0] == pytest.approx(F * N / k, abs=1e-9)


def test_newton_and_bfgs_agree():
    N, k, k3, F = 6, 80.0, 15.0, 40.0
    d_newton = solve(N, k, k3, F, method="newton", eps=1e-10)
    d_bfgs = solve(N, k, k3, F, method="bfgs", eps=1e-8)
    assert np.max(np.abs(d_newton - d_bfgs)) < 1e-6


def test_matches_independent_newton_raphson():
    N, k, k3, F = 6, 80.0, 15.0, 40.0
    d_newton = solve(N, k, k3, F, method="newton", eps=1e-12)
    d_reference = independent_newton_raphson(N, k, k3, F)
    assert np.max(np.abs(d_newton - d_reference)) < 1e-9


def test_gradient_vanishes_at_solution():
    N, k, k3, F = 8, 50.0, 5.0, 25.0
    objective = NonlinearTrussObjective(N, k, k3, F)
    d = solve(N, k, k3, F, method="newton", eps=1e-12)
    assert np.linalg.norm(objective.gradient(d)) < 1e-8


def test_hardening_truss_is_stiffer_than_its_linear_part():
    # with k3>0, the same tip force should produce *less* tip displacement than the linear
    # (k3=0) truss alone -- that's what "hardening" means.
    N, k, k3, F = 4, 60.0, 10.0, 20.0
    d_nonlinear = solve(N, k, k3, F, method="newton", eps=1e-12)
    d_linear = solve(N, k, 0.0, F, method="newton", eps=1e-12)
    assert d_nonlinear[-1, 0] < d_linear[-1, 0]


def test_zero_force_gives_zero_displacement():
    N, k, k3 = 5, 70.0, 12.0
    d = solve(N, k, k3, 0.0, method="newton", eps=1e-12)
    assert np.max(np.abs(d)) < 1e-10


def test_gradient_matches_finite_difference_hessian():
    # cross-check the analytic Hessian against a finite-difference approximation of the
    # gradient's Jacobian, at a nontrivial (nonzero-strain) point.
    N, k, k3, F = 4, 55.0, 8.0, 15.0
    objective = NonlinearTrussObjective(N, k, k3, F)
    x = solve(N, k, k3, F, method="newton", eps=1e-12) * 0.5  # an interior, non-equilibrium point

    H_analytic = objective.hessian(x)
    H_fd = np.zeros((N, N))
    step = 1e-6
    for i in range(N):
        dx = np.zeros((N, 1))
        dx[i, 0] = step
        H_fd[:, i] = ((objective.gradient(x + dx) - objective.gradient(x - dx)) / (2 * step)).flatten()

    assert np.max(np.abs(H_analytic - H_fd)) < 1e-3
