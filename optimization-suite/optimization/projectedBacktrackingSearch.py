# Optimization for Engineers - Dr.Johannes Hild
# projected backtracking line search

# Purpose: Find t to satisfy f(x+t*d)< f(x) - sigma/t*norm(x-P(x - t*gradient))**2

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# P: box projection class with method .project()
# x: column vector in R**n (domain point)
# d: column vector in R**n (search direction)
# sigma: value in (0,1), marks quality of decrease. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
# should return t = 0.5

import numpy as np


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, verbose=0):
    xp = P.project(x)
    gradx = f.gradient(xp)
    decrease = gradx.T @ d

    if decrease >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 1:
        raise TypeError('range of sigma is wrong!')

    if verbose:
        print('Start projectedBacktrackingSearch...')

    beta = 0.5
    t = 1
    maxIter = 100  # guards against t underflowing to 0 (and 1/t blowing up) if xp is already a
                   # stationary point of the box-constrained problem along d
    countIter = 0
    while True:  # Loop until the condition is satisfied

        xp_new = P.project(xp + t * d)
        f_xp_new = f.objective(xp_new)
        norm_term = np.linalg.norm(xp- P.project(xp-t*gradx))**2

        if norm_term == 0 or f_xp_new <= f.objective(xp) - (sigma / t) * norm_term:
            break  # Condition satisfied (or nothing left to decrease against), exit the loop

        t *= beta  # Backtracking step
        countIter += 1
        if countIter > maxIter:
            if verbose:
                print('Warning: iteration limit reached, returning the smallest step size tried.')
            break

    if verbose:
        print('projectedBacktrackingSearch terminated with t=', t)

    return t
