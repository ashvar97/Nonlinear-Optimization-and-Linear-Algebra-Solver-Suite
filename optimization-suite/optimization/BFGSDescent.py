# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = BFGSDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]


import numpy as np
from .WolfePowellSearch import WolfePowellSearch as WP


def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0
    x = x0
    n = x0.shape[0]
    E = np.eye(n)
    B = E
    while True:
        gradx = f.gradient(x)
        d = -np.dot(B, gradx)

        if np.dot(gradx.flatten(), d.flatten()) > 0:  # Not a descent direction
            d = -gradx.flatten()
            B = E

        alpha = WP(f, x, d)

        x_new = x + alpha * d
        s = x_new - x
        x = x_new

        gradx_new = f.gradient(x)
        y = gradx_new - gradx

        if np.linalg.norm(gradx_new) < eps or countIter >= 30:
            break

        curvature = np.dot(y.T, s)  # Curvature condition: must be bounded away from 0
        if np.abs(curvature) > 1.0e-10:
            rho = 1.0 / curvature
            A1 = E - rho * np.dot(s, y.T)
            A2 = E - rho * np.dot(y, s.T)
            B = np.dot(A1, np.dot(B, A2)) + rho * np.dot(s, s.T)
        elif verbose:
            print('Skipping BFGS update: curvature condition y.T@s is too close to 0.')

        countIter += 1

    if verbose:
        print('BFGSDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx), 'and the inverse BFGS matrix is')
        print(B)

    return x
