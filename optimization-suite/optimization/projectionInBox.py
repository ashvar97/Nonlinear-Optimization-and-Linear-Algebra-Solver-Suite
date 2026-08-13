# Optimization for Engineers - Dr.Johannes Hild
# Box projection

# Purpose: Provides .project() and .activeIndexSet() for the box constraint set
# Omega = {x in R**n : a <= x <= b}, as required by projectedBacktrackingSearch,
# projectedBFGSDescent, projectedInexactNewtonCG and augmentedLagrangianDescent.

# Input Definition:
# a: column vector in R**n, elementwise lower bound
# b: column vector in R**n, elementwise upper bound, a <= b required
# eps: nonnegative scalar, tolerance for deciding whether a coordinate is on the boundary
#      (used by activeIndexSet()). Default value: 1.0e-6.

# Output Definition:
# .project(x): column vector in R**n, elementwise clip of x into [a, b]
# .activeIndexSet(x): boolean column vector in R**n, True where the projection of x is
#                      within eps of the lower or upper bound (a "constrained"/active coordinate)

# Required files:
# < none >

# Test cases:
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x = np.array([[3], [0]], dtype=float)
# myBox.project(x) should return [[2], [1]]
# myBox.activeIndexSet(x) should return [[True], [True]]

# x = np.array([[0], [1.5]], dtype=float)
# myBox.project(x) should return [[0], [1.5]]
# myBox.activeIndexSet(x) should return [[False], [False]]

import numpy as np


class projectionInBox:

    def __init__(self, a: np.array, b: np.array, eps=1.0e-6):
        if np.any(a > b):
            raise ValueError('a must be elementwise less than or equal to b.')
        if eps < 0:
            raise ValueError('range of eps is wrong!')

        self.a = a
        self.b = b
        self.eps = eps

    def project(self, x: np.array):
        return np.minimum(np.maximum(x, self.a), self.b)

    def activeIndexSet(self, x: np.array, eps=None):
        if eps is None:
            eps = self.eps

        xp = self.project(x)
        active = (xp - self.a <= eps) | (self.b - xp <= eps)
        return active
