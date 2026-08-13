# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[-0.6], [0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = inexactNewtonCG(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]]


import numpy as np
from .WolfePowellSearch import WolfePowellSearch as WP
from .directionalHessApprox import directionalHessApprox as DHA


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start inexactNewtonCG...')

    countIter = 0
    x = x0.copy()
    gradx = f.gradient(x)
    eta = min(0.5, np.sqrt(np.linalg.norm(gradx))) * np.linalg.norm(gradx)

    while np.linalg.norm(gradx) > eps:
        countIter += 1
        dk = -gradx
        dH = DHA(f, x, dk)
        rho = np.dot(dk.T, dH)

        if rho > eps * np.linalg.norm(dk) ** 2:
            # First CG step
            rj = gradx
            dj = -rj
            xj = x.copy()
            dA = dH
            rhoj = rho
            tj = np.linalg.norm(rj) ** 2 / rhoj
            xj =xj+ tj * dj
            rold = rj
            rj = rold + tj * dA
            betaj = np.linalg.norm(rj) ** 2 / np.linalg.norm(rold) ** 2
            dj = -rj + betaj * dj

            while np.linalg.norm(rj) > eta:
                # Additional CG steps
                dA = DHA(f, x, dj)
                rhoj = np.dot(dj.T, dA)
                tj = np.linalg.norm(rj) ** 2 / rhoj
                xj= xj+ tj * dj
                rold = rj
                rj = rold + tj * dA
                betaj = np.linalg.norm(rj) ** 2 / np.linalg.norm(rold) ** 2
                dj = -rj + betaj * dj

            dk = xj - x

        tk = WP(f, x, dk)
        x =x+ tk * dk
        gradx = f.gradient(x)
        eta = min(0.5, np.sqrt(np.linalg.norm(gradx))) * np.linalg.norm(gradx)

    if verbose:
        gradx = f.gradient(x)
        print('inexactNewtonCG terminated after', countIter, 'steps with norm of gradient =', np.linalg.norm(gradx))

    return x


