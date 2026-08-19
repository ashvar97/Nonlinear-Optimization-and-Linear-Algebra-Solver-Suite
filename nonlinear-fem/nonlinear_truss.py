"""A geometrically-linear but *materially nonlinear* 1D truss, solved by minimizing total
potential energy with optimization-suite's own Newton/BFGS descent -- the one thing this repo's
name promises (nonlinear optimization *and* a linear-algebra/FEM solver suite, working together)
that nothing else in it actually delivers. `fem-truss-beam-solver/` only ever assembles `K*d=F`
and solves it directly, because every element there is linear-elastic; this module is what you
reach for the moment an element's force-displacement law stops being a straight line.

Physics
-------
A chain of N bar elements in series, node 0 fixed, a point force F applied at the free end
(node N). Each element has a cubic-hardening force law instead of Hooke's law:

    force(e) = k*e + k3*e**3            (e = elongation of the element)

with strain-energy density chosen so that its derivative gives exactly that force:

    psi(e) = 0.5*k*e**2 + 0.25*k3*e**4

Total potential energy of the whole chain, as a function of the N free-DOF displacements
d = (d_1, ..., d_N) (d_0 = 0 is the fixed end):

    Pi(d) = sum_i psi(e_i) - F*d_N,   e_i = d_i - d_{i-1}

By the principle of minimum potential energy, the equilibrium displacement is the d that
minimizes Pi -- exactly the problem optimization-suite's descent methods solve. k3=0 recovers
the ordinary linear FEM truss (and this module is checked against the closed-form linear
solution in that limit -- see tests/).
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "optimization-suite"))
from optimization import NewtonDescent, BFGSDescent  # noqa: E402


class NonlinearTrussObjective:
    """Total potential energy of an N-element cubic-hardening 1D truss chain, and its gradient
    and Hessian with respect to the N free-DOF displacements -- the `.objective/.gradient/
    .hessian` interface every optimization-suite descent method expects (see e.g.
    `optimization-suite/optimization/quadraticObjective.py` for the same shape).

    Parameters
    ----------
    N : number of elements (= number of free DOFs; node 0 is fixed)
    k : linear stiffness EA/L of each element
    k3 : cubic hardening coefficient of each element (0 recovers the linear truss)
    F : point force applied at the free end (node N)
    """

    def __init__(self, N: int, k: float, k3: float, F: float):
        self.N = N
        self.k = k
        self.k3 = k3
        self.F = F

    def _elongations(self, x: np.array) -> np.array:
        d = np.concatenate([[[0.0]], x], axis=0).flatten()  # d_0..d_N, d_0 fixed at 0
        return np.diff(d)  # e[i] = d_{i+1} - d_i, element i+1, i = 0..N-1

    def _force(self, e: np.array) -> np.array:
        return self.k * e + self.k3 * e**3

    def _tangent_stiffness(self, e: np.array) -> np.array:
        return self.k + 3 * self.k3 * e**2

    def objective(self, x: np.array) -> float:
        e = self._elongations(x)
        energy = np.sum(0.5 * self.k * e**2 + 0.25 * self.k3 * e**4)
        return energy - self.F * x[-1, 0]

    def gradient(self, x: np.array) -> np.array:
        e = self._elongations(x)
        f = self._force(e)  # f[i] = force in element i+1
        g = np.zeros(self.N)
        for j in range(1, self.N + 1):  # node j
            f_j = f[j - 1]                        # element j pulls on node j with +f_j
            f_jp1 = f[j] if j < self.N else 0.0    # element j+1 pulls on node j with -f_{j+1}
            g[j - 1] = f_j - f_jp1
        g[self.N - 1] -= self.F
        return g.reshape(-1, 1)

    def hessian(self, x: np.array) -> np.array:
        e = self._elongations(x)
        kt = self._tangent_stiffness(e)  # per-element tangent stiffness, size N
        H = np.zeros((self.N, self.N))
        for j in range(1, self.N + 1):
            kt_j = kt[j - 1]
            kt_jp1 = kt[j] if j < self.N else 0.0
            H[j - 1, j - 1] = kt_j + kt_jp1
            if j < self.N:
                H[j - 1, j] = -kt_jp1
                H[j, j - 1] = -kt_jp1
        return H


def solve(N: int, k: float, k3: float, F: float, method: str = "newton",
          eps: float = 1.0e-10, verbose: int = 0) -> np.array:
    """Solve for equilibrium displacements of the N-element chain under tip force F.

    method: "newton" (optimization_suite.NewtonDescent, uses the analytic Hessian above) or
    "bfgs" (optimization_suite.BFGSDescent, gradient-only, quasi-Newton).
    Returns the (N,1) displacement vector d_1..d_N.
    """
    objective = NonlinearTrussObjective(N, k, k3, F)
    x0 = np.zeros((N, 1))
    if method == "newton":
        return NewtonDescent(objective, x0, eps, verbose)
    if method == "bfgs":
        return BFGSDescent(objective, x0, eps, verbose)
    raise ValueError(f"unknown method {method!r}, expected 'newton' or 'bfgs'")
