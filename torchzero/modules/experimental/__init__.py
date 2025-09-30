"""Those are various ideas of mine plus some other modules that I decided not to move to other sub-packages for whatever reason. This is generally less tested and shouldn't be used."""
from .adanystrom import AdaNystrom
from .common_directions_whiten import CommonDirectionsWhiten
from .coordinate_momentum import CoordinateMomentum
from .cubic_adam import CubicAdam
from .curveball import CurveBall
from .fft import FFTProjection
from .gradmin import GradMin
from .higher_order_newton import HigherOrderNewton
from .l_infinity import InfinityNormTrustRegion
from .lmsubspace import (
    LMSubspace,
    SubspaceAdam,
    SubspaceCubicAdam,
    SubspaceFullMatrixAdam,
    SubspaceMomentum,
    SubspaceOptimizerBase,
    SubspaceWhiten,
)
from .newton_solver import NewtonSolver
from .newtonnewton import NewtonNewton
from .reduce_outward_lr import ReduceOutwardLR
from .scipy_newton_cg import ScipyNewtonCG
from .spsa1 import SPSA1
from .structural_projections import BlockPartition, TensorizeProjection

# from dct import DCTProjection
