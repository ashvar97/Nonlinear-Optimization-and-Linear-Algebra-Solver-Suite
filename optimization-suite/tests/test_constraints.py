"""Tests for projectionInBox -- the box-projection module this suite was missing.

Without this class, projectedBFGSDescent, projectedInexactNewtonCG and
augmentedLagrangianDescent could not run at all (see test_descent.py).
"""
import numpy as np

from optimization import projectionInBox


def test_project_clips_outside_points():
    a = np.array([[-2], [1]])
    b = np.array([[2], [2]])
    box = projectionInBox(a, b)
    x = np.array([[3], [0]], dtype=float)
    np.testing.assert_allclose(box.project(x), [[2], [1]])


def test_project_leaves_interior_points_unchanged():
    a = np.array([[-2], [1]])
    b = np.array([[2], [2]])
    box = projectionInBox(a, b)
    x = np.array([[0], [1.5]], dtype=float)
    np.testing.assert_allclose(box.project(x), [[0], [1.5]])


def test_active_index_set_flags_boundary_coordinates():
    a = np.array([[-2], [1]])
    b = np.array([[2], [2]])
    box = projectionInBox(a, b)
    x = np.array([[3], [0]], dtype=float)  # projects to [2, 1] -- both on a bound
    np.testing.assert_array_equal(box.activeIndexSet(x), [[True], [True]])

    x_interior = np.array([[0], [1.5]], dtype=float)
    np.testing.assert_array_equal(box.activeIndexSet(x_interior), [[False], [False]])


def test_rejects_inverted_bounds():
    a = np.array([[2], [2]])
    b = np.array([[-2], [-2]])
    try:
        projectionInBox(a, b)
        assert False, "expected a ValueError when a > b"
    except ValueError:
        pass
