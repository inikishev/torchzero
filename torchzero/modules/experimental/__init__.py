"""Those are various ideas of mine plus some other modules that I decided not to move to other sub-packages for whatever reason. This is generally less tested and shouldn't be used."""
from .curveball import CurveBall

# from dct import DCTProjection
from .fft import FFTProjection
from .gradmin import GradMin
from .l_infinity import InfinityNormTrustRegion
from .momentum import (
    CoordinateMomentum,
    NesterovEMASquared,
    PrecenteredEMASquared,
    SqrtNesterovEMASquared,
)
from .newton_solver import NewtonSolver
from .newtonnewton import NewtonNewton
from .reduce_outward_lr import ReduceOutwardLR
from .scipy_newton_cg import ScipyNewtonCG
from .structural_projections import BlockPartition, TensorizeProjection
