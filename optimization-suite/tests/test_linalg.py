"""Tests for the linear-algebra solvers (incompleteCholesky, LLTSolver, CGSolver, PrecCGSolver).

Every case below is taken directly from the worked examples in each module's docstring.
"""
import numpy as np

from optimization import CGSolver, LLTSolver, PrecCGSolver, incompleteCholesky


def test_llt_solver_small():
    L = np.array([[2, 0, 0], [0.5, np.sqrt(15 / 4), 0], [0, 0, 2]], dtype=float)
    r = np.array([[5], [5], [4]], dtype=float)
    np.testing.assert_allclose(LLTSolver(L, r), [[1], [1], [1]], atol=1e-8)


def test_llt_solver_5x5():
    L = np.array([[22, 0, 0, 0, 0], [17, 13, 0, 0, 0], [13, -2, 17, 0, 0],
                  [8, -4, -7, 18, 0], [4, -5, -4, -5, 19]], dtype=float)
    r = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
    np.testing.assert_allclose(LLTSolver(L, r), [[1], [0], [2], [0], [3]], atol=1e-6)


def test_llt_solver_rejects_zero_diagonal():
    L = np.array([[0, 0], [1, 1]], dtype=float)
    r = np.array([[1], [1]], dtype=float)
    try:
        LLTSolver(L, r)
        assert False, "expected an exception for a zero diagonal element"
    except Exception:
        pass


def test_incomplete_cholesky_negative_delta_is_full_cholesky():
    A = np.array([[5, 4, 3, 2, 1], [4, 5, 2, 1, 0], [3, 2, 5, 0, 0],
                  [2, 1, 0, 5, 0], [1, 0, 0, 0, 5]], dtype=float)
    L = incompleteCholesky(A, alpha=0, delta=-1)
    np.testing.assert_allclose(L @ L.T, A, atol=1e-8)


def test_incomplete_cholesky_alpha_floors_diagonal():
    A = np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]], dtype=float)
    L = incompleteCholesky(A, alpha=1.0e-3, delta=1.0e-6)
    expected = np.array([[2, 0, 0], [0.5, 1.9365, 0], [0, 0, 2]])
    np.testing.assert_allclose(L, expected, atol=1e-3)


def test_incomplete_cholesky_alpha_at_diagonal_bound():
    # alpha == a diagonal entry: the ">" comparison fails, so that entry falls back to sqrt(alpha).
    A = np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]], dtype=float)
    L = incompleteCholesky(A, alpha=4, delta=1.0e-6)
    np.testing.assert_allclose(L, [[2, 0, 0], [0.5, 2, 0], [0, 0, 2]], atol=1e-8)


def test_incomplete_cholesky_large_delta_forces_diagonal_only():
    A = np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]], dtype=float)
    L = incompleteCholesky(A, alpha=1.0e-3, delta=1)
    np.testing.assert_allclose(L, [[2, 0, 0], [0, 2, 0], [0, 0, 2]], atol=1e-8)


def test_incomplete_cholesky_rejects_asymmetric_input():
    A = np.array([[1, 2], [0, 1]], dtype=float)
    try:
        incompleteCholesky(A)
        assert False, "expected a ValueError for a non-symmetric matrix"
    except ValueError:
        pass


def test_cg_solver_3x3():
    A = np.array([[4, 1, 0], [1, 7, 0], [0, 0, 3]], dtype=float)
    b = np.array([[5], [8], [3]], dtype=float)
    np.testing.assert_allclose(CGSolver(A, b, 1.0e-6), [[1], [1], [1]], atol=1e-4)


def test_cg_solver_5x5():
    A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6],
                  [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
    b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
    np.testing.assert_allclose(CGSolver(A, b, 1.0e-6), [[1], [0], [2], [0], [3]], atol=1e-4)


def test_prec_cg_solver_3x3():
    A = np.array([[4, 1, 0], [1, 7, 0], [0, 0, 3]], dtype=float)
    b = np.array([[5], [8], [3]], dtype=float)
    np.testing.assert_allclose(PrecCGSolver(A, b, 1.0e-6), [[1], [1], [1]], atol=1e-6)


def test_prec_cg_solver_5x5():
    A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6],
                  [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
    b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
    np.testing.assert_allclose(PrecCGSolver(A, b, 1.0e-6), [[1], [0], [2], [0], [3]], atol=1e-6)
