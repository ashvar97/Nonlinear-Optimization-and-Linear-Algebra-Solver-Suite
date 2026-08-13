# Optimization for Engineers - Dr.Johannes Hild
# model objective
# Do not change this file

# Nonlinear function R**3 -> R
# test function depending on parameters p in R**3 and mapping
# x -> p[0] * (x[1] + 1) * x[0]**2 + exp(p[1] * x[2] + 1) * x[1]**2 + p[2] * sqrt(|x[0] + 1|) * x[2]**2

# Class parameters:
# p: vector in R**3 (parameter space)

# Input Definition:
# x: vector in R**3 (domain space)

# Output Definition:
# objective(): real number, evaluation at x for parameters p
# gradient(): vector in R**3, evaluation of gradient wrt x
# hessian(): matrix in R**3x3, evaluation of hessian wrt x
# setParameters(): sets p
# parameterGradient(): vector in R**3, evaluation of gradient wrt p

# Required files:
# < none >

# Test cases:
# myObjective = modelObjective([3,2,16]).objective(np.array([[0],[0],[-1 / 2]], dtype=float))
# should return
# myObjective = 4

# myGradient = modelObjective([3,2,16]).gradient(np.array([[0],[0],[-1 / 2]], dtype=float))
# should return
# myGradient = [[2],[0],[-16]]

# myHessian = modelObjective([3,2,16]).hessian(np.array([[0],[0],[-1 / 2]], dtype=float))
# should return
# myHessian = [[5, 0, -8],[0, 2, 0],[-8, 0, 32]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class modelObjective:

    def __init__(self, p: np.array):
        self.p = p

    def objective(self, x: np.array):
        f = self.p[0, 0] * (x[1, 0] + 1) * x[0, 0] ** 2 + np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 \
            + self.p[2, 0] * np.sqrt(np.abs(x[0, 0] + 1)) * x[2, 0] ** 2
        return f

    def gradient(self, x: np.array):
        if x[0, 0] > -1:
            f_dx0 = 2 * self.p[0, 0] * (x[1, 0] + 1) * x[0, 0] + self.p[2, 0] / 2 * x[2, 0] ** 2 / np.sqrt(x[0, 0] + 1)
        elif x[0, 0] == -1:
            raise ValueError('x[0,0] is -1, gradient and Hessian are not defined in this case.')
        else:
            f_dx0 = 2 * self.p[0, 0] * (x[1, 0] + 1) * x[0, 0] - self.p[2, 0] / 2 * x[2, 0] ** 2 / np.sqrt(-x[0, 0] - 1)

        f_dx1 = self.p[0, 0] * x[0, 0] ** 2 + 2 * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0]
        f_dx2 = self.p[1, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 + 2 * self.p[2, 0] * np.sqrt(np.abs(
            x[0, 0] + 1)) * x[2, 0]

        g = np.array([[f_dx0], [f_dx1], [f_dx2]])
        return g

    def hessian(self, x: np.array):
        if x[0, 0] > -1:
            f_dx00 = 2 * self.p[0, 0] * (x[1, 0] + 1) - self.p[2, 0] / 4 * x[2, 0] ** 2 / np.sqrt((x[0, 0] + 1) ** 3)
            f_dx02 = self.p[2, 0] * x[2, 0] / np.sqrt(x[0, 0] + 1)
        elif x[0, 0] == -1:
            raise ValueError('x[0,0] is -1, gradient and Hessian are not defined in this case.')
        else:
            f_dx00 = 2 * self.p[0, 0] * (x[1, 0] + 1) - self.p[2, 0] / 4 * x[2, 0] ** 2 / np.sqrt((-x[0, 0] - 1) ** 3)
            f_dx02 = -self.p[2, 0] * x[2, 0] / np.sqrt(-x[0, 0] - 1)

        f_dx01 = 2 * self.p[0, 0] * x[0, 0]
        f_dx11 = 2 * np.exp(self.p[1, 0] * x[2, 0] + 1)
        f_dx12 = 2 * self.p[1, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0]
        f_dx22 = self.p[1, 0] ** 2 * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2 + 2 * self.p[2, 0] * np.sqrt(np.abs(x[0, 0] + 1))
        h = np.array([[f_dx00, f_dx01, f_dx02], [f_dx01, f_dx11, f_dx12], [f_dx02, f_dx12, f_dx22]])
        return h

    def setParameters(self, p: np.array):
        self.p = p

    def parameterGradient(self, x: np.array):
        R_dp1 = (x[1, 0] + 1) * x[0, 0] ** 2
        R_dp2 = x[2, 0] * np.exp(self.p[1, 0] * x[2, 0] + 1) * x[1, 0] ** 2
        R_dp3 = np.sqrt(x[0, 0] + 1) * x[2, 0] ** 2

        myGradP = np.array([[R_dp1], [R_dp2], [R_dp3]], dtype=float)

        return myGradP

    @staticmethod
    def getXData():
        xdata = np.array([[0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 4.0, 0.0, 8.0, 4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, 4.0, 4.0, -4.0, -4.0, 4.0, 4.0, 0.0, 0.0, 0.0, -4.0, 4.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])
        return xdata

    @staticmethod
    def getFData():
        fdata = np.array([[22.0, -522.0, 22.0, 1014.0, 337.0, -207.0, 337.0, 1329.0, 48.0, 0.0, 192.0, -101.0, 283.0, 84.0, 84.0]])
        return fdata
