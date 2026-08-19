"""Visual demo: nonlinear (cubic-hardening) vs. linear tip response of a truss chain, solved
with optimization-suite's NewtonDescent and BFGSDescent, plus a convergence-iteration comparison.

Run with:
    python examples/demo.py

Produces nonlinear_vs_linear.png in the same directory: a force-vs-tip-displacement curve for
the nonlinear truss against its linear (k3=0) counterpart, showing the hardening effect, plus a
panel comparing how many outer iterations Newton vs. BFGS need to reach the same equilibrium.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nonlinear_truss import NonlinearTrussObjective  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "optimization-suite"))
from optimization import NewtonDescent, BFGSDescent  # noqa: E402


class _IterCounter:
    """Wraps an objective and counts .gradient() calls, as a proxy for outer iterations."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def objective(self, x):
        return self._inner.objective(x)

    def gradient(self, x):
        self.calls += 1
        return self._inner.gradient(x)

    def hessian(self, x):
        return self._inner.hessian(x)


N, k, k3 = 6, 80.0, 15.0
# starts just above 0: at F=0 the truss is already at its (trivial) unstrained equilibrium, so
# the starting point x0=0 has an exactly-zero gradient -- BFGSDescent's line search isn't set up
# to handle a zero-length descent direction at the very first iterate, so we sidestep that
# degenerate corner case rather than exercise it here.
forces = np.linspace(0.5, 100, 25)

tip_nonlinear, tip_linear = [], []
newton_iters, bfgs_iters = [], []

for F in forces:
    obj_nl = NonlinearTrussObjective(N, k, k3, F)
    obj_lin = NonlinearTrussObjective(N, k, 0.0, F)

    x0 = np.zeros((N, 1))
    d_nl = NewtonDescent(obj_nl, x0, 1e-10)
    d_lin = NewtonDescent(obj_lin, x0, 1e-10)
    tip_nonlinear.append(d_nl[-1, 0])
    tip_linear.append(d_lin[-1, 0])

    counted_newton = _IterCounter(obj_nl)
    NewtonDescent(counted_newton, x0, 1e-8)
    counted_bfgs = _IterCounter(obj_nl)
    BFGSDescent(counted_bfgs, x0, 1e-8)
    newton_iters.append(counted_newton.calls)
    bfgs_iters.append(counted_bfgs.calls)

print(f"Verification: at F={forces[-1]:.1f}, nonlinear tip disp = {tip_nonlinear[-1]:.4f}, "
      f"linear tip disp = {tip_linear[-1]:.4f} "
      f"({'nonlinear is stiffer, as expected for hardening' if tip_nonlinear[-1] < tip_linear[-1] else 'UNEXPECTED'})")

fig, (ax_curve, ax_iters) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_curve.plot(tip_linear, forces, "b--", linewidth=2, label="linear (k3=0)")
ax_curve.plot(tip_nonlinear, forces, "r", linewidth=2, label="cubic-hardening (k3>0)")
ax_curve.set_xlabel("tip displacement")
ax_curve.set_ylabel("tip force F")
ax_curve.set_title(f"{N}-element truss: force vs. tip displacement")
ax_curve.legend()

ax_iters.plot(forces, newton_iters, "o-", label="NewtonDescent")
ax_iters.plot(forces, bfgs_iters, "s-", label="BFGSDescent")
ax_iters.set_xlabel("tip force F")
ax_iters.set_ylabel("gradient evaluations to converge")
ax_iters.set_title("Convergence cost: Newton vs. BFGS")
ax_iters.legend()

fig.tight_layout()
out_path = Path(__file__).resolve().parent / "nonlinear_vs_linear.png"
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")
