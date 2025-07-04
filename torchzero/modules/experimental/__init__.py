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
from .eigendescent import EigenDescent
from .etf import (
    ExponentialTrajectoryFit,
    ExponentialTrajectoryFitV2,
    PointwiseExponential,
)
from .exp_adam import ExpAdam
from .gradmin import GradMin
from .newton_solver import NewtonSolver
from .reduce_outward_lr import ReduceOutwardLR
from .structured_newton import StructuredNewton
from .subspace_preconditioners import (
    HistorySubspacePreconditioning,
    RandomSubspacePreconditioning,
)
from .tensor_adagrad import TensorAdagrad
