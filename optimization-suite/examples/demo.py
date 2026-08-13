"""Visual demo: Newton vs. BFGS descent on the banana valley, and the effect of a box
constraint on projectedBFGSDescent.

Run with:
    python examples/demo.py

Produces banana_valley_descent.png in the same directory: a contour plot of the classic
Rosenbrock-style "banana valley" test function with the iterate path of each method overlaid,
plus a second panel showing how projectedBFGSDescent's path bends to respect a box constraint
that the unconstrained optimum lies outside of.
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization import (
    BFGSDescent, NewtonDescent, bananaValleyObjective, projectedBFGSDescent, projectionInBox,
    simpleValleyObjective,
)


class _TrajectoryRecorder:
    """Wraps an objective and records every point its .gradient() is evaluated at.

    Each descent method here computes the gradient exactly once per outer iterate, so the
    recorded points trace out (an approximation of) the path taken to the minimum -- without
    needing to modify any of the library's descent implementations.
    """

    def __init__(self, objective):
        self._objective = objective
        self.visited = []

    def objective(self, x):
        return self._objective.objective(x)

    def gradient(self, x):
        self.visited.append(np.array(x, dtype=float).ravel())
        return self._objective.gradient(x)

    def hessian(self, x):
        return self._objective.hessian(x)


def _banana_grid():
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 100 * (X2 - X1 ** 2) ** 2 + (1 - X1) ** 2 + 2
    return X1, X2, Z


def run_trajectory(method, x0, objective_factory=bananaValleyObjective, **kwargs):
    recorder = _TrajectoryRecorder(objective_factory())
    xmin = method(recorder, x0, **kwargs)
    path = np.array(recorder.visited + [np.array(xmin, dtype=float).ravel()])
    return xmin, path


def main():
    out_dir = Path(__file__).parent
    X1, X2, Z = _banana_grid()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: unconstrained Newton vs. BFGS ---
    x0 = np.array([[-1.5], [-0.5]], dtype=float)
    _, newton_path = run_trajectory(NewtonDescent, x0, eps=1.0e-6)
    _, bfgs_path = run_trajectory(BFGSDescent, x0, eps=1.0e-6)

    ax1.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 25), cmap="Greys", linewidths=0.6)
    ax1.plot(*newton_path.T, "o-", color="tab:blue", label=f"NewtonDescent ({len(newton_path)} steps)")
    ax1.plot(*bfgs_path.T, "s-", color="tab:orange", markersize=3, linewidth=0.8,
             label=f"BFGSDescent ({len(bfgs_path)} steps, incl. one wild first step off-frame)")
    ax1.plot(1, 1, "*", color="black", markersize=16, label="global minimum")
    ax1.plot(*x0.ravel(), "kx", markersize=10, label="start")
    # BFGS's first step (identity initial Hessian, steep banana-valley wall) overshoots far
    # outside this window before the curvature estimate corrects it -- clip the view to the
    # region that actually shows the interesting behavior near the minimum.
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1, 3)
    ax1.set_title("Unconstrained descent on the banana valley")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(fontsize=7, loc="upper left")

    # --- Panel 2: box-constrained projectedBFGSDescent on a well-conditioned valley ---
    # (simpleValleyObjective, not the banana valley: projectedBFGSDescent's line search can take
    # a very long time to crawl along a box edge on a badly-conditioned objective -- exactly the
    # kind of case its curvature-condition guard now degrades gracefully on rather than crashing,
    # but it's still not a fast demo, so we exercise it here on the well-behaved objective its
    # own documented test cases use.)
    p = np.array([[1], [1]])
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-0.5, 2.5, 400)
    X1v, X2v = np.meshgrid(x1, x2)
    Zv = np.cosh(X1v) + p[0, 0] * (X2v - 1) ** 2 + p[1, 0]

    a = np.array([[-10], [-10]])
    b = np.array([[10], [0.5]])  # excludes the unconstrained minimum (0, 1) via the x2 bound only
    box = projectionInBox(a, b)
    x0b = np.array([[1.2], [1.8]], dtype=float)
    _, proj_path = run_trajectory(lambda f, x, **kw: projectedBFGSDescent(f, box, x, **kw), x0b,
                                   objective_factory=lambda: simpleValleyObjective(p), eps=1.0e-6)

    ax2.contour(X1v, X2v, Zv, levels=np.linspace(Zv.min(), Zv.min() + 8, 20), cmap="Greys", linewidths=0.6)
    ax2.add_patch(plt.Rectangle((a[0, 0], a[1, 0]), b[0, 0] - a[0, 0], b[1, 0] - a[1, 0],
                                 fill=True, color="tab:green", alpha=0.15, label="feasible box"))
    ax2.plot(*proj_path.T, "o-", color="tab:red", label=f"projectedBFGSDescent ({len(proj_path)} steps)")
    ax2.plot(0, 1, "*", color="black", markersize=16, label="unconstrained minimum (infeasible)")
    ax2.plot(*x0b.ravel(), "kx", markersize=10, label="start")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_title("Box-constrained descent stays feasible")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = out_dir / "banana_valley_descent.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"NewtonDescent took {len(newton_path) - 1} steps, BFGSDescent took {len(bfgs_path) - 1} steps.")
    print(f"projectedBFGSDescent converged to {proj_path[-1]} (pinned to the box's edge at x2={b[1, 0]}).")


if __name__ == "__main__":
    main()
