r"""
Ready to use optimizers.

:any:`torchzero.optim.modular.ModularOptimizer` - build a custom optimizer using modules.

:any:`torchzero.optim.quasi_newton` - quasi-newton 2nd order optimizers.

:any:`torchzero.optim.zeroth_order` - zeroth order optimizers.

:any:`torchzero.optim.second_order` - exact 2nd order optimizers.

"""
from .modular import ModularOptimizer
from .quasi_newton import NewtonGradFDM, NewtonRaySearch, DiagNewtonRaySearch
from .zeroth_order import *
from .second_order import ExactNewton