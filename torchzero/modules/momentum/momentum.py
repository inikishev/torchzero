from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from .ema import EMA


class HeavyBall(EMA):
    """Polyak's momentum (heavy-ball method).

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        debiased (bool, optional): whether to debias the EMA like in Adam. Defaults to False.
        lerp (bool, optional):
            whether to use linear interpolation, if True, this becomes exponential moving average. Defaults to False.
        ema_init (str, optional): initial values for the EMA, "zeros" or "update".
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, debiased: bool = False, lerp=False, ema_init: Literal['zeros', 'update'] = 'update', target: Target = 'update'):
        super().__init__(momentum=momentum, dampening=dampening, debiased=debiased, lerp=lerp, ema_init=ema_init, target=target)

def nag_(
    tensors_: TensorList,
    velocity_: TensorList,
    momentum: float | NumberList,
    dampening: float | NumberList,
    lerp: bool = False,
):
    """Nesterov momentum.

    Returns `tensors_`"""
    if lerp: velocity_.lerp_(tensors_, 1 - momentum)
    else: velocity_.add_(tensors_).mul_(momentum)

    tensors_ += velocity_.lazy_mul(1 - dampening)

    return tensors_


class NAG(Transform):
    """Nesterov accelerated gradient method (nesterov momentum).

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        lerp (bool, optional):
            whether to use linear interpolation, if True, this becomes similar to exponential moving average. Defaults to False.
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, lerp=False, target: Target = 'update'):
        defaults = dict(momentum=momentum,dampening=dampening, lerp=lerp)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        velocity = unpack_states(states, tensors, 'velocity', cls=TensorList)
        lerp = self.settings[params[0]]['lerp']

        momentum,dampening = unpack_dicts(settings, 'momentum','dampening', cls=NumberList)
        return nag_(TensorList(tensors), velocity_=velocity,momentum=momentum,dampening=dampening,lerp=lerp)

