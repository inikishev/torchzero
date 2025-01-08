from typing import Literal, Unpack

from ...core import OptimizerModule
from ...modules import DirectionalNewton as _DirectionalNewton
from ...modules import (_CommonKwargs, _get_baked_in_and_module_lr,
                        _make_common_modules)
from ..modular import Modular


class DirectionalNewton(Modular):
    """Minimizes a parabola in the direction of the gradient via one additional forward pass,
    and uses another forward pass to make sure it didn't overstep.
    So in total this performs three forward passes and one backward.

    This can only be used either as the first module or after FDM, as it requires ascent to
    be the gradient.

    First forward and backward pass is used to calculate the value and gradient at initial parameters.
    Then a gradient descent step is performed with `lr` learning rate, and loss is recalculated
    with new parameters. A quadratic is fitted to two points and gradient,
    if it has positive curvature, this makes a step towards the minimum, and checks if lr decreased
    with an additional forward pass.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional):
            learning rate. Since you shouldn't put this module after LR(), you have to specify
            the learning rate in this argument. Defaults to 1e-2.
        max_dist (float | None, optional):
            maximum distance to step when minimizing quadratic.
            If minimum is further than this distance, minimization is not performed. Defaults to 1e4.
        validate_step (bool, optional):
            uses an additional forward pass to check
            if step towards the minimum actually decreased the loss. Defaults to True.
        log_lrs (bool, optional):
            saves lrs and losses with them into optimizer._lrs (for debugging).
            Defaults to False.

    Note:
        While lr scheduling is supported, this uses lr of the first parameter for all parameters.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        max_dist: float | None = 1e5,
        validate_step: bool = True,

    ):

        modules = _DirectionalNewton(lr, max_dist, validate_step)
        super().__init__(params, modules)

