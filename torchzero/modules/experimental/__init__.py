"""Those are various ideas of mine plus some other modules that I decided not to move to other sub-packages for whatever reason. This is generally less tested and shouldn't be used."""
from .absoap import ABSOAP
from .adadam import Adadam
from .adam_abs import AdamAbs
from .adam_exp import AdamExp
from .adam_expclogmul import AdamExpclogmul
from .adam_lambertw import AdamLambertW
from .adam_sqrt import AdamSqrt
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
from .expanded_lbfgs import ExpandedLBFGS
from .fft import FFTProjection
from .gradmin import GradMin
from .hnewton import HNewton
from .modular_lbfgs import ModularLBFGS
from .momentum import (
    CoordinateMomentum,
    NesterovEMASquared,
    PrecenteredEMASquared,
    SqrtNesterovEMASquared,
)
from .newton_solver import NewtonSolver
from .newtonnewton import NewtonNewton
from .parabolic_search import CubicParabolaSearch, ParabolaSearch
from .polyss import PolyStepSize
from .reduce_outward_lr import ReduceOutwardLR
from .scipy_newton_cg import ScipyNewtonCG
from .structural_projections import BlockPartition, TensorizeProjection
from .subspace_preconditioners import (
    HistorySubspacePreconditioning,
    RandomSubspacePreconditioning,
)
