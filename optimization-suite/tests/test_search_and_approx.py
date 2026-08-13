"""Tests for line searches and derivative approximations."""
import numpy as np

from optimization import (
    SUCSimplexGradient, SUCStencilFailure, directionalHessApprox, noHessianObjective,
    projectedBacktrackingSearch, projectedHessApprox, projectionInBox,
    simpleValleyObjective, WolfePowellSearch,
)


def test_wolfe_powell_case1():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    x = np.array([[-1.01], [1]])
    d = np.array([[1], [1]])
    assert WolfePowellSearch(myObjective, x, d, 1.0e-3, 1.0e-2) == 1


def test_wolfe_powell_case2():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    x = np.array([[-1.2], [1]])
    d = np.array([[0.1], [1]])
    assert WolfePowellSearch(myObjective, x, d, 1.0e-3, 1.0e-2) == 16


def test_wolfe_powell_case3():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    x = np.array([[-0.2], [1]])
    d = np.array([[1], [1]])
    assert WolfePowellSearch(myObjective, x, d, 1.0e-3, 1.0e-2) == 0.25


def test_wolfe_powell_rejects_ascent_direction():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    x = np.array([[-1.01], [1]])
    d = np.array([[-1], [-1]])  # not a descent direction here
    try:
        WolfePowellSearch(myObjective, x, d)
        assert False, "expected a TypeError for a non-descent direction"
    except TypeError:
        pass


def test_projected_backtracking_search():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    box = projectionInBox(np.array([[-2], [1]]), np.array([[2], [2]]), 1.0e-6)
    x = np.array([[1], [1]])
    d = np.array([[-1.99], [0]])
    assert projectedBacktrackingSearch(myObjective, box, x, d, 0.5) == 0.5


def test_directional_hess_approx():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    x = np.array([[-1.01], [1]])
    d = np.array([[1], [1]])
    np.testing.assert_allclose(directionalHessApprox(myObjective, x, d), [[1.55491], [0]], atol=1e-4)


def test_projected_hess_approx_matches_unconstrained_when_interior():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    box = projectionInBox(np.array([[-2], [-2]]), np.array([[2], [2]]))
    x = np.array([[-1.01], [1]])
    d = np.array([[1], [1]])
    unconstrained = directionalHessApprox(myObjective, x, d)
    projected = projectedHessApprox(myObjective, box, x, d)
    np.testing.assert_allclose(projected, unconstrained, atol=1e-9)


def test_projected_hess_approx_zeroes_active_coordinates():
    myObjective = simpleValleyObjective(np.array([[0], [1]]))
    box = projectionInBox(np.array([[-2], [-2]]), np.array([[2], [2]]))
    x = np.array([[2.0], [1.0]])  # coordinate 0 sits on the upper bound: active
    d = np.array([[1], [1]])
    projected = projectedHessApprox(myObjective, box, x, d)
    assert projected[0, 0] == 0.0


def test_suc_simplex_gradient_near_stationary_point():
    x = np.array([[-0.015793], [0.012647]], dtype=float)
    grad = SUCSimplexGradient(noHessianObjective, x, 1.0e-6)
    np.testing.assert_allclose(grad, [[0], [0]], atol=1e-2)


def test_suc_stencil_failure_at_a_minimum():
    # At (an approximation of) the global minimum, no stencil point should decrease the
    # objective, so a stencil failure (1) should be reported.
    x_min = np.array([[0.261], [-0.209]], dtype=float)
    assert SUCStencilFailure(noHessianObjective, x_min, 1.0e-3) == 1


def test_suc_stencil_failure_at_a_maximum_finds_descent():
    # At the local maximum, every coordinate direction is a descent direction, so this should
    # NOT report a stencil failure.
    x_max = np.array([[-0.015793], [0.012647]], dtype=float)
    assert SUCStencilFailure(noHessianObjective, x_max, 1.0e-3) == 0
