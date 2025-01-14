r"""
This submodule contains composable optimizer "building blocks".
"""

from ..core.module import OptimizerModule
from . import experimental
from .adaptive import *
from .gradient_approximation import *
from .line_search import *
from .meta import *
from .misc import *
from .momentum import *
from .operations import *
from .optimizers import *
from .orthogonalization import *
from .quasi_newton import *
from .regularization import *
from .scheduling import *
from .second_order import *
from .smoothing import *
from .weight_averaging import *
