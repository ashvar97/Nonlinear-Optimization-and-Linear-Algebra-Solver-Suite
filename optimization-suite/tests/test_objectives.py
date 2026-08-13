"""Tests for the test-function / objective classes, from their docstring examples."""
import numpy as np

from optimization import (
    augmentedLagrangianObjective, bananaValleyObjective, leastSquaresModel,
    leastSquaresObjective, modelObjective, noHessianObjective, quadraticObjective,
    simpleValleyObjective,
)


def test_banana_valley():
    x = np.array([[1], [1]], dtype=float)
    assert bananaValleyObjective.objective(x) == 2
    np.testing.assert_allclose(bananaValleyObjective.gradient(x), [[0], [0]], atol=1e-8)
    np.testing.assert_allclose(bananaValleyObjective.hessian(x), [[802, -400], [-400, 200]])


def test_quadratic_objective():
    A = np.eye(2)
    b = np.ones((2, 1))
    q = quadraticObjective(A, b, 1)
    assert q.objective(b) == 4
    np.testing.assert_allclose(q.gradient(b), [[2], [2]])
    np.testing.assert_allclose(q.hessian(b), [[1, 0], [0, 1]])


def test_simple_valley_objective():
    p = np.array([[1], [2]])
    x = np.array([[0], [1]])
    sv = simpleValleyObjective(p)
    assert sv.objective(x) == 3
    np.testing.assert_allclose(sv.gradient(x), [[0], [0]])
    np.testing.assert_allclose(sv.hessian(x), [[1, 0], [0, 2]])
    np.testing.assert_allclose(sv.parameterGradient(x), [[0], [1]])


def test_model_objective():
    mo = modelObjective(np.array([[3], [2], [16]]))
    x = np.array([[0], [0], [-1 / 2]], dtype=float)
    assert mo.objective(x) == 4
    np.testing.assert_allclose(mo.gradient(x), [[2], [0], [-16]])
    np.testing.assert_allclose(mo.hessian(x), [[5, 0, -8], [0, 2, 0], [-8, 0, 32]])


def test_no_hessian_objective_at_local_max():
    x = np.array([[-0.015793], [0.012647]], dtype=float)
    np.testing.assert_allclose(noHessianObjective.objective(x), 3.0925, atol=1e-3)
    np.testing.assert_allclose(noHessianObjective.gradient(x), [[0], [0]], atol=1e-2)


def test_least_squares_objective():
    p0 = np.array([[2], [3]])
    model = simpleValleyObjective(p0)
    xData = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    fData = np.array([[2, 3, 2.54, 4.76]])
    lso = leastSquaresObjective(model, xData, fData)
    np.testing.assert_allclose(lso.residual(p0), [[2], [3], [10], [20]], atol=1e-2)
    np.testing.assert_allclose(lso.jacobian(p0), [[0, 1], [1, 1], [4, 1], [9, 1]])


def test_least_squares_model():
    p0 = np.array([[2], [3]])
    model = simpleValleyObjective(p0)
    xData = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    fData = np.array([[2, 3, 2.54, 4.76]])
    lsm = leastSquaresModel(model, xData, fData)
    np.testing.assert_allclose(lsm.residual(p0), [[2], [3], [10], [20]], atol=1e-2)
    np.testing.assert_allclose(lsm.jacobian(p0), [[0, 1], [1, 1], [4, 1], [9, 1]])


def test_augmented_lagrangian_objective():
    f = quadraticObjective(np.array([[2, 0], [0, 2]], dtype=float), np.zeros((2, 1)), 1)
    h = quadraticObjective(np.array([[2, 0], [0, 2]], dtype=float), np.zeros((2, 1)), -1)
    x0 = np.array([[2], [2]], dtype=float)
    al = augmentedLagrangianObjective(f, h, alpha=-1, gamma=10)
    np.testing.assert_allclose(al.objective(x0), 247, atol=1e-8)
    np.testing.assert_allclose(al.gradient(x0), [[280], [280]], atol=1e-8)


def test_augmented_lagrangian_objective_rejects_nonpositive_gamma():
    f = quadraticObjective(np.eye(2), np.zeros((2, 1)), 0)
    h = quadraticObjective(np.eye(2), np.zeros((2, 1)), -1)
    try:
        augmentedLagrangianObjective(f, h, alpha=0, gamma=0)
        assert False, "expected a ValueError for gamma <= 0"
    except ValueError:
        pass
