# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
from .projectedBacktrackingSearch import projectedBacktrackingSearch as PB
from .projectedHessApprox import projectedHessApprox as PHA


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    eta_k = np.min([1/2, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))])

    while np.linalg.norm(xp - P.project(xp - f.gradient(xp))) > eps:
        countIter += 1


        x_j = xp.copy()
        r_j = f.gradient(xp)
        d_j = -r_j

    
        while np.linalg.norm(r_j) > eta_k:
 
            d_A = PHA(f, P, xp, d_j)

            rho_j = np.dot(d_j.T, d_A)

            if rho_j <= eps * np.linalg.norm(d_j)**2:
                break
      
            t_j = np.linalg.norm(r_j)**2 / rho_j
            x_j += t_j * d_j

            r_old = r_j.copy()
            r_j = r_old + t_j * d_A
   
            beta_j = np.linalg.norm(r_j)**2 / np.linalg.norm(r_old)**2
            d_j = -r_j + beta_j * d_j

        # c) Set d_k←x_j-x_k, but only if the loop did not break due to curvature fail at the very first try.
        # In that case, set d_k←-∇f(x_k ).
        if rho_j > eps * np.linalg.norm(d_j)**2:
            d_k = x_j - xp
        else:
            d_k = -f.gradient(xp)


        t_k = PB(f, P, xp, d_k)

    
        xp = P.project(xp + t_k * d_k)
        eta_k = np.min([1/2, np.sqrt(np.linalg.norm(xp - P.project(xp - f.gradient(xp)))) * np.linalg.norm(xp - P.project(xp - f.gradient(xp)))])


    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp
