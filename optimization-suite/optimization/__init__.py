"""optimization: a small nonlinear-optimization and linear-algebra solver suite.

Originally written for the "Optimization for Engineers" course (Dr. Johannes Hild); see
../README.md for background, algorithm overview and usage examples.
"""

# Linear algebra
from .incompleteCholesky import incompleteCholesky
from .LLTSolver import LLTSolver
from .CGSolver import CGSolver
from .PrecCGSolver import PrecCGSolver

# Test / example objectives
from .bananaValleyObjective import bananaValleyObjective
from .quadraticObjective import quadraticObjective
from .simpleValleyObjective import simpleValleyObjective
from .modelObjective import modelObjective
from .noHessianObjective import noHessianObjective
from .leastSquaresObjective import leastSquaresObjective
from .leastSquaresModel import leastSquaresModel
from .augmentedLagrangianObjective import augmentedLagrangianObjective

# Constraints
from .projectionInBox import projectionInBox

# Line search / derivative approximation
from .WolfePowellSearch import WolfePowellSearch
from .projectedBacktrackingSearch import projectedBacktrackingSearch
from .directionalHessApprox import directionalHessApprox
from .projectedHessApprox import projectedHessApprox
from .SUCSimplexGradient import SUCSimplexGradient, SUCStencilFailure

# Descent methods
from .NewtonDescent import NewtonDescent
from .inexactNewtonCG import inexactNewtonCG
from .projectedInexactNewtonCG import projectedInexactNewtonCG
from .BFGSDescent import BFGSDescent
from .projectedBFGSDescent import projectedBFGSDescent
from .levenbergMarquardtDescent import levenbergMarquardtDescent
from .implicitFiltering import implicitFiltering, derivativefreeBFGSDescent
from .augmentedLagrangianDescent import augmentedLagrangianDescent

__all__ = [
    "incompleteCholesky", "LLTSolver", "CGSolver", "PrecCGSolver",
    "bananaValleyObjective", "quadraticObjective", "simpleValleyObjective",
    "modelObjective", "noHessianObjective", "leastSquaresObjective", "leastSquaresModel",
    "augmentedLagrangianObjective",
    "projectionInBox",
    "WolfePowellSearch", "projectedBacktrackingSearch",
    "directionalHessApprox", "projectedHessApprox",
    "SUCSimplexGradient", "SUCStencilFailure",
    "NewtonDescent", "inexactNewtonCG", "projectedInexactNewtonCG",
    "BFGSDescent", "projectedBFGSDescent", "levenbergMarquardtDescent",
    "implicitFiltering", "derivativefreeBFGSDescent", "augmentedLagrangianDescent",
]
