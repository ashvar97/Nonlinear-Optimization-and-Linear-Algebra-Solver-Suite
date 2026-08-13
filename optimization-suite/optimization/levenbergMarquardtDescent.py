# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# p0 = np.array([[180],[0]])
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[0, 0], [1, 2]])
# fk = np.array([[2, 3]])
# myErrorVector = leastSquaresModel(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[1], [1]]

import numpy as np
from .PrecCGSolver import PrecCGSolver as PCG


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0:
        raise TypeError('range of alpha0 is wrong!')

    if beta <= 1:
        raise TypeError('range of beta is wrong!')

    if verbose:
        print('Start levenbergMarquardtDescent...')

    countIter = 0
    p = p0.copy().astype(float)
    anew = alpha0

    while np.linalg.norm(R.jacobian(p).T @ R.residual(p)) > eps:
        jacobian_p = R.jacobian(p)
        residual_p = R.residual(p)
        jac_transpose_res_matrix=-jacobian_p.T @ residual_p
        jac_transpose_jac_matrix=jacobian_p.T @ jacobian_p + anew * np.eye(len(p))
        dk = PCG(jac_transpose_jac_matrix, jac_transpose_res_matrix)
        fpdk=0.5 * np.linalg.norm(R.residual(p + dk))**2
        fp=0.5 * np.linalg.norm(residual_p)**2
        if fpdk < fp:
            p += dk
            anew = alpha0
        else:
            anew *= beta
        countIter += 1
    if verbose:
        print('levenbergMarquardtDescent terminated after', countIter, 'steps with norm of gradient =',
              np.linalg.norm(R.jacobian(p).T @ R.residual(p)))
    return p
