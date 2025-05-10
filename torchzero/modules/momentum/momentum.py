from typing import Literal

import torch

from ...core import Target, Transform
from ...utils import NumberList, TensorList
from .ema import EMA


class HeavyBall(EMA):
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
    def __init__(self, momentum:float=0.9, dampening:float=0, lerp=False, target: Target = 'update'):
        defaults = dict(momentum=momentum,dampening=dampening, lerp=lerp)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        velocity = self.get_state('velocity', params=params, cls=TensorList)
        lerp = self.settings[params[0]]['lerp']

        momentum,dampening = self.get_settings('momentum','dampening', params=params, cls=NumberList)
        return nag_(TensorList(tensors), velocity_=velocity,momentum=momentum,dampening=dampening,lerp=lerp)
