import numpy as np

from .incompleteCholesky import incompleteCholesky as IC
from .LLTSolver import LLTSolver as LLT


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start PrecCGSolver...')

    countIter = 0

    L = IC(A)
    x = LLT(L, b)
    r = A @ x - b
    z = np.linalg.solve(L.T, np.linalg.solve(L, r))  # Preconditioning: z = L^(-T) L^(-1) r
    p = z  # Initial search direction
    rsold = np.dot(r.T, z)  # For computing beta
    while np.linalg.norm(r) > delta:
        countIter += 1
        if countIter > len(b):
            if verbose:
                print("Iteration limit reached.")
            break
        Ap = A @ p
        alpha = rsold / np.dot(p.T, Ap)
        x = x - alpha * p
        r = r - alpha * Ap
        z = np.linalg.solve(L.T, np.linalg.solve(L, r))  # Preconditioning
        rsnew = np.dot(r.T, z)
        beta = rsnew / rsold
        p = z + beta * p
        rsold = rsnew

    if verbose:
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r))

    return x