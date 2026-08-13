"""Tests for the descent methods -- the heart of the suite.

projectedBFGSDescent, projectedInexactNewtonCG and augmentedLagrangianDescent were entirely
unrunnable before projectionInBox and projectedHessApprox were implemented (both raised
ModuleNotFoundError on import). The tests here are the regression guard for that fix.
"""
import numpy as np

from optimization import (
    BFGSDescent, NewtonDescent, augmentedLagrangianDescent, bananaValleyObjective,
    implicitFiltering, inexactNewtonCG, leastSquaresModel, levenbergMarquardtDescent,
    noHessianObjective, projectedBFGSDescent, projectedInexactNewtonCG, projectionInBox,
    quadraticObjective, simpleValleyObjective,
)


def test_newton_descent_on_banana_valley():
    x0 = np.array([[0], [1]], dtype=float)
    xmin = NewtonDescent(bananaValleyObjective(), x0, 1.0e-6)
    np.testing.assert_allclose(xmin, [[1], [1]], atol=1e-4)


def test_newton_descent_rejects_bad_eps():
    try:
        NewtonDescent(bananaValleyObjective(), np.zeros((2, 1)), eps=0)
        assert False, "expected a TypeError for eps <= 0"
    except TypeError:
        pass


def test_inexact_newton_cg_finds_global_minimum():
    x0 = np.array([[-0.01], [0.01]], dtype=float)
    xmin = inexactNewtonCG(noHessianObjective, x0, 1.0e-6)
    np.testing.assert_allclose(xmin, [[0.26], [-0.21]], atol=5e-2)


def test_bfgs_descent_finds_global_minimum():
    x0 = np.array([[-0.01], [0.01]], dtype=float)
    xmin = BFGSDescent(noHessianObjective, x0, 1.0e-6)
    np.testing.assert_allclose(xmin, [[0.26], [-0.21]], atol=5e-2)


def test_levenberg_marquardt_fits_simple_valley_parameters():
    p0 = np.array([[180], [0]], dtype=float)
    model = simpleValleyObjective(p0)
    xData = np.array([[0, 0], [1, 2]])
    fData = np.array([[2, 3]])
    errorVector = leastSquaresModel(model, xData, fData)
    pmin = levenbergMarquardtDescent(errorVector, p0, 1.0e-4, 1.0e-3, 100)
    np.testing.assert_allclose(pmin, [[1], [1]], atol=5e-2)


def test_implicit_filtering_finds_a_stationary_point():
    # implicitFiltering is derivative-free; check it lands near one of noHessianObjective's
    # two documented minima rather than pinning an exact target.
    x0 = np.array([[0.3], [-0.3]], dtype=float)
    h = np.array([[0.1], [0.01], [0.001]], dtype=float)
    xmin = implicitFiltering(noHessianObjective, x0, h, 1.0e-3)
    global_min = np.array([[0.261], [-0.209]])
    local_min = np.array([[-0.265], [0.212]])
    close_to_global = np.allclose(xmin, global_min, atol=5e-2)
    close_to_local = np.allclose(xmin, local_min, atol=5e-2)
    assert close_to_global or close_to_local


# --- Previously-broken (projection-dependent) algorithms ---

def test_projected_bfgs_descent_interior_start():
    myObjective = simpleValleyObjective(np.array([[1], [1]]))
    box = projectionInBox(np.array([[1], [1]]), np.array([[2], [2]]))
    x0 = np.array([[2], [2]], dtype=float)
    xmin = projectedBFGSDescent(myObjective, box, x0, 1.0e-3)
    np.testing.assert_allclose(xmin, [[1], [1]], atol=5e-2)


def test_projected_bfgs_descent_banana_valley_converges_quickly():
    myObjective = bananaValleyObjective()
    box = projectionInBox(np.array([[-10], [-10]]), np.array([[10], [10]]))
    x0 = np.array([[0], [1]], dtype=float)
    xmin = projectedBFGSDescent(myObjective, box, x0, 1.0e-6)
    np.testing.assert_allclose(xmin, [[1], [1]], atol=1e-3)


def test_projected_inexact_newton_cg():
    myObjective = simpleValleyObjective(np.array([[1], [1]]))
    box = projectionInBox(np.array([[1], [1]]), np.array([[2], [2]]))
    x0 = np.array([[2], [2]], dtype=float)
    xmin = projectedInexactNewtonCG(myObjective, box, x0, 1.0e-3)
    np.testing.assert_allclose(xmin, [[1], [1]], atol=5e-2)


def test_augmented_lagrangian_descent_equality_constrained_quadratic():
    f = quadraticObjective(np.array([[4, 0], [0, 2]], dtype=float), np.zeros((2, 1)), 1)
    box = projectionInBox(np.zeros((2, 1)), np.full((2, 1), 2))
    h = quadraticObjective(np.array([[2, 0], [0, 2]], dtype=float), np.zeros((2, 1)), -1)
    x0 = np.array([[1], [1]], dtype=float)
    xmin, alphamin = augmentedLagrangianDescent(f, box, h, x0, alpha0=0, eps=1.0e-3, delta=1.0e-6)
    np.testing.assert_allclose(xmin, [[0], [1]], atol=5e-2)
    np.testing.assert_allclose(alphamin, -1, atol=5e-2)
