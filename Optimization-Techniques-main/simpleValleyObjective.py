# Optimization for Engineers - Dr.Johannes Hild
# Simple valley objective
# Do not change this file

# test function depending on parameter vector  p and mapping
# x -> cosh(x(0)) + p(0) * (x(1) - 1) ** 2 + p(1)

# Class parameters:
# p: vector in R**2 (parameter space)

# Input Definition:
# x: vector in R**2 (domain space)

# Output Definition:
# objective: real number, evaluation at x for parameters p
# gradient: vector in R**2, evaluation of gradient wrt x
# hessian: matrix in R**2x2, evaluation of hessian wrt x
# setParameters(): sets p
# parameterGradient(): vector in R**2, evaluation of gradient wrt p

# Required files:
# < none >

# Test cases:
# p = np.array([[1],[2]])
# x = np.array([[0],[1]])
# myObjective = simpleValleyObjective(p).objective(x)
# should return
# myObjective = 3

# myGradient = simpleValleyObjective(p).gradient(x)
# should return
# myGradient = [[0],[0]]

# myHessian = simpleValleyObjective(p).hessian(x)
# should return
# myHessian = [[1, 0],[0, 2]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class simpleValleyObjective:

    def __init__(self, p: np.array):
        self.p = p

    def objective(self, x: np.array):
        p0 = self.p[0, 0]
        p1 = self.p[1, 0]
        x0 = x[0, 0]
        x1 = x[1, 0]
        f = np.cosh(x0) + p0*(x1-1)**2 + p1
        return f

    def gradient(self, x: np.array):
        p0 = self.p[0, 0]
        x0 = x[0, 0]
        x1 = x[1, 0]
        f_dx0 = np.sinh(x0)
        f_dx1 = 2*p0*(x1-1)
        g = np.array([[f_dx0], [f_dx1]])
        return g

    def hessian(self, x: np.array):
        p0 = self.p[0, 0]
        x0 = x[0, 0]
        f_dx00 = np.cosh(x0)
        f_dx01 = 0
        f_dx11 = 2 * p0
        h = np.array([[f_dx00, f_dx01], [f_dx01, f_dx11]])
        return h

    def setParameters(self, p: np.array):
        self.p = p

    @staticmethod
    def parameterGradient(x: np.array):
        myGradP = np.array([[(x[1, 0] - 1)**2], [1]], dtype=float)

        return myGradP
