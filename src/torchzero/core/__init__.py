import sys

from .module import (
    OptimizationState,
    OptimizerModule,
    _Chain,
    _Chainable,
    _get_loss,
    _ScalarLoss,
    _Targets,
)

if sys.version_info[1] < 12:
    from .tensorlist_optimizer311 import TensorListOptimizer, ParamsT, _ClosureType, _maybe_pass_backward
else:
    from .tensorlist_optimizer import TensorListOptimizer, ParamsT, _ClosureType, _maybe_pass_backward