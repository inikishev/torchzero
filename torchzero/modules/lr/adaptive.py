"""Various step size strategies"""
import random
from typing import Any
from operator import itemgetter
import torch

from ...core import Transform
from ...utils import TensorList, NumberList, unpack_dicts


class PolyakStepSize(Transform):
    """Polyak's step-size method.

    Args:
        max (float | None, optional): maximum possible step size. Defaults to None.
        min_obj_value (int, optional):
            (estimated) minimal possible value of the objective function (lowest possible loss). Defaults to 0.
        use_grad (bool, optional):
            if True, uses dot product of update and gradient to compute the step size.
            Otherwise, dot product of update with itself is used, which has no geometric meaning so it probably won't work well.
            Defaults to True.
        parameterwise (bool, optional):
            if True, calculate Polyak step-size for each parameter separately,
            if False calculate one global step size for all parameters. Defaults to False.
        alpha (float, optional): multiplier to Polyak step-size. Defaults to 1.
    """
    def __init__(self, max: float | None = None, min_obj_value: float = 0, use_grad=True, parameterwise=False, alpha: float = 1):

        defaults = dict(alpha=alpha, max=max, min_obj_value=min_obj_value, use_grad=use_grad, parameterwise=parameterwise)
        super().__init__(defaults, uses_grad=use_grad)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        tensors = TensorList(tensors)
        grads = TensorList(grads)
        alpha = NumberList(s['alpha'] for s in settings)

        parameterwise, use_grad, max, min_obj_value = itemgetter('parameterwise', 'use_grad', 'max', 'min_obj_value')(settings[0])

        if use_grad: denom = tensors.dot(grads)
        else: denom = tensors.dot(tensors)

        if parameterwise:
            polyak_step_size: TensorList | Any = (loss - min_obj_value) / denom.where(denom!=0, 1)
            polyak_step_size = polyak_step_size.where(denom != 0, 0)
            if max is not None: polyak_step_size = polyak_step_size.clamp_max(max)

        else:
            if denom.abs() <= torch.finfo(denom.dtype).eps: polyak_step_size = 0 # converged
            else: polyak_step_size = (loss - min_obj_value) / denom

            if max is not None:
                if polyak_step_size > max: polyak_step_size = max

        tensors.mul_(alpha * polyak_step_size)
        return tensors


class RandomStepSize(Transform):
    """Uses random global or layer-wise step size from `low` to `high`.

    Args:
        low (float, optional): minimum learning rate. Defaults to 0.
        high (float, optional): maximum learning rate. Defaults to 1.
        parameterwise (bool, optional):
            if True, generate random step size for each parameter separately,
            if False generate one global random step size. Defaults to False.
    """
    def __init__(self, low: float = 0, high: float = 1, parameterwise=False, seed:int|None=None):
        defaults = dict(low=low, high=high, parameterwise=parameterwise,seed=seed)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        s = settings[0]
        parameterwise = s['parameterwise']

        seed = s['seed']
        if 'generator' not in self.global_state:
            self.global_state['generator'] = random.Random(seed)
        generator: random.Random = self.global_state['generator']

        if parameterwise:
            low, high = unpack_dicts(settings, 'low', 'high')
            lr = [generator.uniform(l, h) for l, h in zip(low, high)]
        else:
            low = s['low']
            high = s['high']
            lr = generator.uniform(low, high)

        torch._foreach_mul_(tensors, lr)
        return tensors
