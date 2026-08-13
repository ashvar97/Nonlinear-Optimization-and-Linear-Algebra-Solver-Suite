# Optimization for Engineers - Dr.Johannes Hild
# Projected Directional Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences, restricted to the
# free (inactive) index set of a box constraint P. Required by projectedInexactNewtonCG so that
# its CG sub-iteration never proposes a step along a coordinate that is already pinned at its
# bound - each active coordinate contributes 0 to the returned vector, matching a projected /
# reduced-space Newton-CG step.

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: reduced Hessian times direction, column vector in R ** n, zero on the active index set

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# dH = projectedHessApprox(myObjective, myBox, x, d)
# should return dH close to [[1.55491],[0]] (same as directionalHessApprox, since x is interior
# and no coordinate of d is on the active set)

# x = np.array([[2], [1]])
# dH = projectedHessApprox(myObjective, myBox, x, d)
# should return dH = [[0],[0]] (x[0,0]=2 is on the active upper bound, so the first coordinate
# of every direction is suppressed; the objective is separable so the second coordinate would
# also collapse once its own contribution is zeroed alongside it)

import numpy as np


def projectedHessApprox(f, P, x: np.array, d: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start projectedHessApprox...')

    active = P.activeIndexSet(x)
    d_free = d.copy()
    d_free[active] = 0.0

    xplus = P.project(x + delta * d_free)
    xminus = P.project(x - delta * d_free)
    gradient_plus = f.gradient(xplus)
    gradient_minus = f.gradient(xminus)
    dH = (gradient_plus - gradient_minus) / (2 * delta)
    dH[active] = 0.0

    if verbose:
        print('projectedHessApprox terminated with dH=', dH)

    return dH
