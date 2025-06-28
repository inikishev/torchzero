"""This submodule contains various untested experimental modules, some of them are to be moved out of experimental when properly tested, some are to remain here forever or to be deleted depending on the degree of their usefulness."""
from .absoap import ABSOAP
from .adadam import Adadam
from .adamY import AdamY
from .adasoap import AdaSOAP
from .curveball import CurveBall
from .eigendescent import EigenDescent
from .etf import (
    ExponentialTrajectoryFit,
    ExponentialTrajectoryFitV2,
    PointwiseExponential,
)
from .gradmin import GradMin
from .newton_solver import NewtonSolver
from .reduce_outward_lr import ReduceOutwardLR
from .structured_newton import StructuredNewton
from .subspace_preconditioners import (
    HistorySubspacePreconditioning,
    RandomSubspacePreconditioning,
)
from .tensor_adagrad import TensorAdagrad
from .higher_order_adagrad import HigherOrderAdagrad
from .cubic_adam import CubicAdam
from .cosine import CosineTrustRegion, CosineDebounce
from .adaptive_step_size import AdaptiveStepSize