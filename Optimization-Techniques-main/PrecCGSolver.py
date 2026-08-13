import numpy as np
import incompleteCholesky as IC
import LLTSolver as LLT


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start PrecCGSolver...')

    countIter = 0

    L = IC.incompleteCholesky(A)
    x = LLT.LLTSolver(L, b)
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


# Test cases:
A1 = np.array([[4, 1, 0], [1, 7, 0], [0, 0, 3]], dtype=float)
b1 = np.array([[5], [8], [3]], dtype=float)
delta1 = 1.0e-6
x1 = PrecCGSolver(A1, b1, delta1, 1)
print("Test case 1:", x1)

A2 = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
b2 = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
delta2 = 1.0e-6
x2 = PrecCGSolver(A2, b2, delta2, 1)
print("Test case 2:", x2)