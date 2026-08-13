# Optimization for Engineers - Dr.Johannes Hild
# Incomplete Cholesky decomposition

# Purpose: incompleteCholesky finds lower triangle matrix L such that A - L * L ^ T is small, but
# eigenvalues are positive and sparsity is preserved

# Input Definition:
# A: real valued symmetric matrix nxn
# alpha: nonnegative scalar, lower bound for eigenvalues of L * L ^ T.Default value: 1.0e-3.
# delta: scalar, if positive it is tolerance for recognizing nonsparse entry.
# If negative, do complete cholesky.Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# A: real valued lower triangle matrix nxn

# Required files:
# < none >

# Test cases:
# alpha = 0
# delta = -1
# verbose = true
# L = incompleteCholesky(np.array([[5, 4, 3, 2, 1],[4, 5, 2, 1, 0],\
# [3, 2, 5, 0, 0],[2, 1, 0, 5, 0],[1, 0, 0, 0, 5]]), alpha, delta, verbose)
# executes complete Cholesky decomposition with norm of residual approx. 8.89e-16
# the warning is okay.

# alpha = 1.0e-3
# delta = 1.0e-6
# verbose = true
# L = incompleteCholesky(np.array([4, 1, 0],[1, 4, 0],[0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0.5 1.94 0], [ 0 0 2]]

# alpha = 4
# delta = 1.0e-6
# verbose = true
# L = incompleteCholesky(np.array([4, 1, 0], [1, 4, 0], [0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0.5 2 0], [ 0 0 2]]

# alpha = 1.0e-3
# delta = 1
# verbose = true
# L = incompleteCholesky(np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]]), alpha, delta, verbose)
# should return approximately
# L = [[2 0 0],[0 2 0], [ 0 0 2]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def incompleteCholesky(A: np.array, alpha=1.0e-3, delta=1.0e-6, verbose=0):
    L = np.copy(A)
    dim = np.shape(L)
    n = dim[0]
    if n != dim[1]:
        raise ValueError('A has wrong dimension.')

    if np.max(np.abs(A-A.T) > 1.0e-6):
        raise ValueError('A is not symmetric.')

    if alpha < 0:
        raise ValueError('range of alpha is wrong!')

    if delta < 0:
        print('Warning: negative delta detected, sparsity is not preserved.')

    if verbose:
        print('Start incompleteCholesky...')

    sqrt_alpha = np.sqrt(alpha)
    for k in range(n):
        if L[k, k] > alpha:
            L[k, k] = np.sqrt(L[k, k])
        else:
            L[k, k] = sqrt_alpha

        for i in range(k+1, n):
            if np.abs(L[i, k]) > delta:
                L[i, k] = L[i, k] / L[k, k]
            else:
                L[i, k] = 0

        for j in range(k+1, n):
            for i in range(j, n):
                if np.abs(L[i, j]) > delta:
                    L[i, j] = L[i, j] - L[i, k] * L[j, k]
            L[k, j] = 0

    if verbose:
        residualmatrix = A - L @ L.T
        residual = np.max(np.abs(residualmatrix))
        print('IncompleteCholesky terminated with norm of residual:')
        print(residual)

    return L
