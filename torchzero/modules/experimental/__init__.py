"""This submodule contains various untested experimental modules, some of them are to be moved out of experimental when properly tested, some are to remain here forever or to be deleted depending on the degree of their usefulness."""
from .absoap import ABSOAP
from .adadam import Adadam
from .adam_lambertw import AdamLambertW
from .adamY import AdamY
from .adaptive_step_size import AdaptiveStepSize
from .adasoap import AdaSOAP
from .cosine import (
    AdaptiveDifference,
    AdaptiveDifferenceEMA,
    CosineDebounce,
    CosineMomentum,
    CosineStepSize,
    ScaledAdaptiveDifference,
)
from .cubic_adam import CubicAdam
from .curveball import CurveBall

# from dct import DCTProjection
from .eigendescent import EigenDescent
from .etf import (
    ExponentialTrajectoryFit,
    ExponentialTrajectoryFitV2,
    PointwiseExponential,
)
from .exp_adam import ExpAdam
from .expanded_lbfgs import ExpandedLBFGS
from .fft import FFTProjection
from .gradmin import GradMin
from .hnewton import HNewton
from .modular_lbfgs import ModularLBFGS
from .newton_solver import NewtonSolver
from .newtonnewton import NewtonNewton
from .parabolic_search import CubicParabolaSearch, ParabolaSearch
from .reduce_outward_lr import ReduceOutwardLR
from .structural_projections import BlockPartition, TensorizeProjection
from .subspace_preconditioners import (
    HistorySubspacePreconditioning,
    RandomSubspacePreconditioning,
)
from .tensor_adagrad import TensorAdagrad
