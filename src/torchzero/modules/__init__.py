r"""
This submodule contains composable optimizer "building blocks".
"""

from .gradient_approximation import FDM, NewtonFDM, RandomizedFDM
from .line_search import *
from .meta import (Chain, ClosureOptimizerWrapper, OptimizerWrapper,
                   UninitializedClosureOptimizerWrapper)
from .momentum import (NesterovMomentum, PolyakMomentum,
                       RandomCoordinateMomentum)
from .operators import *
from .quasi_newton import NewtonGradFDM
from .regularization import AddNoise, WeightDecay
from .second_order import ExactNewton, LinearSystemSolvers, FallbackLinearSystemSolvers
from .optimizers import Adam, SGD
from .smoothing import LaplacianSmoothing
from .subspace import (Proj2Masks, ProjAscent, ProjAscentRay, ProjGrad,
                       ProjGradRay, ProjRandom, Subspace)
