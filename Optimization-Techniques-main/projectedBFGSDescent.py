# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
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
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0.1], [0.1]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[1.5], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[-0.26],[0.21]] (if it is close to [[0.26],[-0.21]] then maybe your reduction is done wrongly)

# myObjective = bananaValleyObjective()
# a = np.array([[-10], [-10]])
# b = np.array([[10], [10]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0], [1]], dtype=float)
# eps = 1.0e-6
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]] in less than 30 iterations. If you have too much iterations, then maybe the hessian is used wrongly.


import numpy as np
import projectedBacktrackingSearch as PB


def matrnr():
    # set your matriculation number here
    matrnr = 23115420
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedBFGSDescent...')

    countIter = 0
    xp = P.project(x0)
    gradx = f.gradient(xp)
    n = len(xp)
    B = np.eye(n)  # Initialize the inverse BFGS matrix as identity

    while np.linalg.norm(xp - P.project(xp - gradx)) > eps:
        d = -np.dot(B, gradx)  # Compute the BFGS direction

        if np.dot(d.T, gradx) >= 0:  # Check if d is a descent direction
            d = -gradx  # If not, set d to the steepest descent direction
            B = np.eye(n)  # Reset the inverse BFGS matrix to identity

        t = PB.projectedBacktrackingSearch(f, P, xp, d)  # Compute the step size
        xp_new = P.project(xp + t * d)  # Perform the projection

        s = xp_new - xp  # Compute the update in x
        y = f.gradient(xp_new) - gradx  # Compute the update in gradient

        rho = 1 / np.dot(y.T, s)  # Compute the scaling factor

        B = (np.eye(n) - rho * np.dot(s, y.T)).dot(B).dot(np.eye(n) - rho * np.dot(y, s.T)) + rho * np.dot(s, s.T)  # Update the inverse BFGS matrix

        xp = xp_new
        gradx = f.gradient(xp)
        countIter += 1

    if verbose:
        print('projectedBFGSDescent terminated after', countIter, 'steps with norm of stationarity =',
              np.linalg.norm(xp - P.project(xp - gradx)))

    return xp
