from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import  TensorTransform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..opt_utils import debias as _debias, ema_


class EMA(TensorTransform):
    """Maintains an exponential moving average of update.

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        debias (bool, optional): whether to debias the EMA like in Adam. Defaults to False.
        lerp (bool, optional): whether to use linear interpolation. Defaults to True.
        ema_init (str, optional): initial values for the EMA, "zeros" or "update".
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, debias: bool = False, lerp=True, ema_init: Literal['zeros', 'update'] = 'zeros'):
        defaults = dict(momentum=momentum,dampening=dampening,debias=debias,lerp=lerp,ema_init=ema_init)
        super().__init__(defaults, uses_grad=False)

        self.add_projected_keys("grad", "exp_avg")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        step = self.global_state['step'] = self.global_state.get('step', 0) + 1

        debias, lerp, ema_init = itemgetter('debias','lerp','ema_init')(settings[0])

        exp_avg = unpack_states(states, tensors, 'exp_avg',
                                init=torch.zeros_like if ema_init=='zeros' else tensors, cls=TensorList)
        momentum, dampening = unpack_dicts(settings, 'momentum','dampening', cls=NumberList)

        exp_avg = ema_(TensorList(tensors), exp_avg_=exp_avg,beta=momentum,dampening=dampening,lerp=lerp)

        if debias: return _debias(exp_avg, step=step, beta1=momentum, alpha=1, inplace=False)
        else: return exp_avg.clone() # this has exp_avg storage so needs to be cloned



class HeavyBall(EMA):
    """Polyak's momentum (heavy-ball method).

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        debias (bool, optional): whether to debias the EMA like in Adam. Defaults to False.
        lerp (bool, optional):
            whether to use linear interpolation, if True, this becomes exponential moving average. Defaults to False.
        ema_init (str, optional): initial values for the EMA, "zeros" or "update".
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, debias: bool = False, lerp=False, ema_init: Literal['zeros', 'update'] = 'update'):
        super().__init__(momentum=momentum, dampening=dampening, debias=debias, lerp=lerp, ema_init=ema_init)

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


class NAG(TensorTransform):
    """Nesterov accelerated gradient method (nesterov momentum).

    Args:
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        dampening (float, optional): momentum dampening. Defaults to 0.
        lerp (bool, optional):
            whether to use linear interpolation, if True, this becomes similar to exponential moving average. Defaults to False.
        target (Target, optional): target to apply EMA to. Defaults to 'update'.
    """
    def __init__(self, momentum:float=0.9, dampening:float=0, lerp=False):
        defaults = dict(momentum=momentum,dampening=dampening, lerp=lerp)
        super().__init__(defaults, uses_grad=False)

        self.add_projected_keys("grad", "velocity")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        velocity = unpack_states(states, tensors, 'velocity', cls=TensorList)
        lerp = self.settings[params[0]]['lerp']

        momentum,dampening = unpack_dicts(settings, 'momentum','dampening', cls=NumberList)
        return nag_(TensorList(tensors), velocity_=velocity,momentum=momentum,dampening=dampening,lerp=lerp)

