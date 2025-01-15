import random
from typing import Any

from ...core import OptimizerModule
from ...tensorlist import TensorList


class PolyakStepSize(OptimizerModule):
    """Polyak step-size. Meant to be used at the beginning when ascent is the gradient but other placements may work.
    This can also work with SGD as SPS (Stochastic Polyak Step-Size) seems to use the same formula.

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

        defaults = dict(alpha = alpha)
        super().__init__(defaults)
        self.max = max
        self.min_obj_value = min_obj_value
        self.use_grad = use_grad
        self.parameterwise = parameterwise

    def _update(self, state, ascent):
        if state.closure is None: raise ValueError("PolyakStepSize requires closure")
        if state.fx0 is None: state.fx0 = state.closure(False) # can only happen when placed after SPSA

        lr = self.get_group_key('lr')

        if self.parameterwise:
            if self.use_grad: denom = (ascent*state.maybe_compute_grad_(self.get_params())).mean()
            else: denom = ascent.pow(2).mean()
            polyak_step_size: TensorList | Any = (state.fx0 - self.min_obj_value) / denom.where(denom!=0, 1) # type:ignore
            polyak_step_size = polyak_step_size.where(denom != 0, 0)
            if self.max is not None: polyak_step_size = polyak_step_size.clamp_max(self.max)

        else:
            if self.use_grad: denom = (ascent*state.maybe_compute_grad_(self.get_params())).total_mean()
            else: denom = ascent.pow(2).total_mean()
            if denom == 0: polyak_step_size = 0 # we converged
            else: polyak_step_size = (state.fx0 - self.min_obj_value) / denom

            if self.max is not None:
                if polyak_step_size > self.max: polyak_step_size = self.max

        ascent.mul_(lr * polyak_step_size)
        return ascent



class RandomStepSize(OptimizerModule):
    """Uses random global step size from `low` to `high`.

    Args:
        low (float, optional): minimum learning rate. Defaults to 0.
        high (float, optional): maximum learning rate. Defaults to 1.
        parameterwise (bool, optional):
            if True, generate random step size for each parameter separately,
            if False generate one global random step size. Defaults to False.
    """
    def __init__(self, low: float = 0, high: float = 1, parameterwise=False):
        super().__init__({})
        self.low = low; self.high = high
        self.parameterwise = parameterwise

    def _update(self, state, ascent):
        if self.parameterwise:
            lr = [random.uniform(self.low, self.high) for _ in range(len(ascent))]
        else:
            lr = random.uniform(self.low, self.high)
        return ascent.mul_(lr) # type:ignore
