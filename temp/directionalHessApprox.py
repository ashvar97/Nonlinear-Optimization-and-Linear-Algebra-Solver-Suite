# Optimization for Engineers - Dr.Johannes Hild
# Directional Hessian Approximation

# Purpose: Approximates Hessian times direction with central differences

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# delta: tolerance for termination. Default value: 1.0e-6
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# dH: Hessian times direction, column vector in R ** n

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])

# dH = directionalHessApprox(myObjective, x, d)
# should return dH = [[1.55491],[0]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23115420
    return matrnr


def directionalHessApprox(f, x: np.array, d: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start directionalHessApprox...')

    # Central difference approximation of directional Hessian
    xplus = x + delta * d
    xminus = x - delta * d
    gradient_plus = f.gradient(xplus)
    gradient_minus = f.gradient(xminus)
    dH = (gradient_plus - gradient_minus) / (2 * delta)

    if verbose:
        print('directionalHessApprox terminated with dH=', dH)

    return dH

