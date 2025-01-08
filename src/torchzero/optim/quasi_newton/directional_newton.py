from ...modules import (
    SGD,
)
from ...modules import DirectionalNewton as _DirectionalNewton
from ..modular import Modular


class DirectionalNewton(Modular):
    """Minimizes a parabola in the direction of the gradient (or update if momentum or weight decay is enabled)
    via one additional forward pass, and uses another forward pass to make sure it didn't overstep.
    So in total this performs three forward passes and one backward.

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
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.

    Note:
        While lr scheduling is supported, this uses lr of the first parameter for all parameters.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        max_dist: float | None = 1e5,
        validate_step: bool = True,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,

    ):

        modules = [
            SGD(1, momentum=momentum,dampening=dampening,weight_decay=weight_decay,nesterov=nesterov),
            _DirectionalNewton(lr, max_dist, validate_step)
        ]
        super().__init__(params, modules)

