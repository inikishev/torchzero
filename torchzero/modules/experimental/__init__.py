"""Those are various ideas of mine plus some other modules that I decided not to move to other sub-packages for whatever reason. This is generally less tested."""
from .coordinate_momentum import CoordinateMomentum
from .cubic_adam import CubicAdam, SubspaceCubicAdam
from .curveball import CurveBall

# from dct import DCTProjection
from .fft import FFTProjection
from .gradmin import GradMin
from .higher_order_newton import HigherOrderNewton
from .l_infinity import InfinityNormTrustRegion
from .newton_solver import NewtonSolver
from .newtonnewton import NewtonNewton
from .reduce_outward_lr import ReduceOutwardLR
from .scipy_newton_cg import ScipyNewtonCG
from .structural_projections import BlockPartition, TensorizeProjection
