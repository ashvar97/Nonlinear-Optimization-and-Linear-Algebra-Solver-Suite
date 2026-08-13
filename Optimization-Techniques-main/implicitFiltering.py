# Optimization for Engineers - Dr.Johannes Hild
# implicit Filtering

# Purpose: Inner and outer loop with inverse BFGS update to find the LMP at all scales of a noisy objective.

# Input Definition:
# f: objective class with method .objective(), can have noise
# x0: column vector in R ** n (domain point), starting point
# h: column vector in R ** m, scales for filtering
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# xmin: column vector in R**n (LMP at all scales)

# Required files:
# grad_f_h = SUCSimplexGradient(f, x, h) from SUCSimplexGradient.py
# isStencilFailure = SUCStencilFailure(f, x, h) from SUCSimplexGradient.py

# Test cases:
# myObjective = noisyObjective()
# x0 = np.array([[1],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
# h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
# xmin = implicitFiltering(myObjective, x0)
# should return xmin close to [[1.027],[0],[0],[0],[0],[0],[0],[0]]

import numpy as np
import SUCSimplexGradient as SUC


def matrnr():
    # set your matriculation number here
    matrnr = 23115420
    return matrnr


def implicitFiltering(f, x0: np.array, h: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start implicitFiltering...')

    isLMPAtAllScales = 0
    xmin = x0
    m = h.shape[0]
    countIter = 0

    xbest = xmin.copy()
    fbest = f.objective(xbest)

    while not isLMPAtAllScales:
        for k in range(m):
            xh = derivativefreeBFGSDescent(f, xmin, h[k], eps, 1.0e-4, verbose)
            if not np.array_equal(xh, xmin, True):
                if f.objective(xh) < fbest:
                    xbest = xh.copy()
                    fbest = f.objective(xbest)
                    if verbose:
                        print('Found descent on scale ', h[k])

        if np.array_equal(xbest, xmin, True):
            isLMPAtAllScales = 1

        xmin = xbest.copy()
        countIter += 1

        if verbose:
            print('STEP ', countIter, ': objective = ', fbest)

    if verbose:
        print('implicitFiltering terminated after ', countIter, ' outer loops with LMP at all scales = ', xmin)
    return xmin


def derivativefreeBFGSDescent(f, x0: np.array, h, eps=1.0e-3, sigma=1.0e-4, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start derivativefreeBFGSDescent...')

    x = x0
    grad_f_h = SUC.SUCSimplexGradient(f, x, h)
    n = x0.shape[0]
    E = np.eye(n)
    B = E

    isStencilFailure = SUC.SUCStencilFailure(f, x, h)
    loopCounter = 0
    linesearchFail = 0

    if isStencilFailure or np.linalg.norm(grad_f_h) <= eps*h or loopCounter > 200*n or linesearchFail:
        satisfiesTermination = 1
    else:
        satisfiesTermination = 0

    while not satisfiesTermination:
        beta_k = min(1, 10 * h / np.linalg.norm(B @ grad_f_h))
        d = -beta_k * B @ grad_f_h

        t = 1
        linesearchCounter = 0
        while f.objective(x + t * d) > f.objective(x) + sigma * t * grad_f_h.T @ d:
            t = 0.5 * t
            linesearchCounter += 1
            if linesearchCounter > 10:
                linesearchFail = 1
                break

        if linesearchFail:
            break
        else:
            x_new = x + t * d
            grad_f_h_new = SUC.SUCSimplexGradient(f, x_new, h)
            delta_gk = grad_f_h_new - grad_f_h
            delta_xk = t * d

            x = x_new
            grad_f_h = grad_f_h_new

            s_k = delta_xk
            y_k = delta_gk
            rho_k = 1 / (y_k.T @ s_k)

            B = (E - rho_k * s_k @ y_k.T) @ B @ (E - rho_k * y_k @ s_k.T) + rho_k * s_k @ s_k.T

            loopCounter += 1

            if isStencilFailure or np.linalg.norm(grad_f_h) <= eps * h or loopCounter > 200 * n or linesearchFail:
                satisfiesTermination = 1
            else:
                satisfiesTermination = 0

    if verbose:
        print('derivativefreeBFGSDescent terminated after', loopCounter, 'steps with norm of gradient =', np.linalg.norm(grad_f_h))

    return x
