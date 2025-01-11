r"""
Ready to use optimizers.
"""
from .modular import Modular
from .quasi_newton import *
from .zeroth_order import *
from .second_order import *
from .first_order import *
from .wrappers.scipy import ScipyMinimize
from . import experimental