# Optimization for Engineers - Dr.Johannes Hild
# LLT Solver

# Purpose: LLTSolver solves  (L @ L.T)*y=r for y using forward and backward substitution

# Input Definition:
# L: real valued lower triangle matrix nxn with nonzero diagonal elements
# r: column vector in R ** n
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# y: column vector in R ** n (solution in domain space)

# Required files:
# < none >

# Test cases:
# L = np.array([[2, 0, 0], [0.5, np.sqrt(15 / 4), 0], [0, 0, 2]], dtype=float)
# r = np.array([[5], [5], [4]], dtype=float)
# y = LLTSolver(L,r)
# should return y = [[1], [1], [1]]

# L = np.array([[22, 0, 0, 0, 0], [17, 13, 0, 0, 0], [13, -2, 17, 0, 0], [8, -4, -7, 18, 0], [4, -5, -4, -5, 19]], dtype=float)
# r = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# y = LLTSolver(L,r)
# should return y = [[1],[0],[2],[0],[3]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def LLTSolver(L: np.array, r: np.array, verbose=0):

    if verbose:
        print('Start LLTSolver...')

    n = np.size(r)
    s = r.copy()
    for i in range(n):
        for j in range(i):
            s[i, 0] = s[i, 0] - L[i, j] * s[j, 0]

        if L[i, i] == 0:
            raise Exception('Zero diagonal element detected...')

        s[i, 0] = s[i, 0] / L[i, i]

    y = s

    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            y[i, 0] = y[i, 0] - L[j, i] * y[j, 0]

        y[i, 0] = y[i, 0] / L[i, i]

    if verbose:
        residual = (L@L.T)@y-r
        print('LLTSolver terminated with residual:')
        print(residual)
    return y
