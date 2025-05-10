import random
from typing import Any

import torch

from ...core import Transform
from ...utils import TensorList, NumberList


class PolyakStepSize(Transform):
    """Polyak step-size.

    Args:
        max (float | None, optional): maximum possible step size. Defaults to None.
        min_obj_value (int, optional): (estimated) minimal possible value of the objective function (lowest possible loss). Defaults to 0.
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
    def transform(self, tensors, params, grads, vars):
        loss = vars.get_loss(False)
        assert grads is not None
        tensors = TensorList(tensors)
        grads = TensorList(grads)
        alpha = self.get_settings('alpha', params=params, cls=NumberList)
        settings = self.settings[params[0]]
        parameterwise = settings['parameterwise']
        use_grad = settings['use_grad']
        max = settings['max']
        min_obj_value = settings['min_obj_value']

        if parameterwise:
            if use_grad: denom = (tensors * grads).sum()
            else: denom = tensors.pow(2).sum()
            polyak_step_size: TensorList | Any = (loss - min_obj_value) / denom.where(denom!=0, 1)
            polyak_step_size = polyak_step_size.where(denom != 0, 0)
            if max is not None: polyak_step_size = polyak_step_size.clamp_max(max)

        else:
            if use_grad: denom = tensors.dot(grads)
            else: denom = tensors.dot(tensors)
            if denom == 0: polyak_step_size = 0 # we converged
            else: polyak_step_size = (loss - min_obj_value) / denom

            if max is not None:
                if polyak_step_size > max: polyak_step_size = max

        tensors.mul_(alpha * polyak_step_size)
        return tensors



class RandomStepSize(Transform):
    """Uses random global step size from `low` to `high`.

    Args:
        low (float, optional): minimum learning rate. Defaults to 0.
        high (float, optional): maximum learning rate. Defaults to 1.
        parameterwise (bool, optional):
            if True, generate random step size for each parameter separately,
            if False generate one global random step size. Defaults to False.
    """
    def __init__(self, low: float = 0, high: float = 1, parameterwise=False):
        defaults = dict(low=low, high=high, parameterwise=parameterwise)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        settings = self.settings[params[0]]
        parameterwise = settings['parameterwise']

        if parameterwise:
            low, high = self.get_settings('low', 'high', params=params)
            lr = [random.uniform(l, h) for l, h in zip(low, high)]
        else:
            low = settings['low']
            high = settings['high']
            lr = random.uniform(low, high)

        torch._foreach_mul_(tensors, lr)
        return tensors
