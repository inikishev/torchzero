import math
import random
import warnings
from functools import partial
from typing import Literal

import numpy as np
import torch

from ...core import Module
from ...utils import NumberList, TensorList

class CD(Module):
    """Coordinate descent. Proposes a descent direction along a single coordinate.
    A line search such as ``tz.m.ScipyMinimizeScalar(maxiter=8)`` or a fixed step size can be used after this.

    Args:
        h (float, optional): finite difference step size. Defaults to 1e-3.
        grad (bool, optional):
            if True, scales direction by gradient estimate. If False, the scale is fixed to 1. Defaults to True.
        adaptive (bool, optional):
            whether to adapt finite difference step size, this requires an additional buffer. Defaults to True.
        index (str, optional):
            index selection strategy.
            - "cyclic" - repeatedly cycles through each coordinate, e.g. ``1,2,3,1,2,3,...``.
            - "cyclic2" - cycles forward and then backward, e.g ``1,2,3,3,2,1,1,2,3,...`` (default).
            - "random" - picks coordinate randomly.
        threepoint (bool, optional):
            whether to use three points (three function evaluatins) to determine descent direction.
            if False, uses two points, but then ``adaptive`` can't be used. Defaults to True.
    """
    def __init__(self, h:float=1e-3, grad:bool=False, adaptive:bool=True, index:Literal['cyclic', 'cyclic2', 'random']="cyclic2", threepoint:bool=True,):
        defaults = dict(h=h, grad=grad, adaptive=adaptive, index=index, threepoint=threepoint)
        super().__init__(defaults)

    def update(self, objective): raise RuntimeError
    def apply(self, objective): raise RuntimeError

    @torch.no_grad
    def step(self, objective):
        closure = objective.closure
        if closure is None:
            raise RuntimeError("CD requires closure")

        params = TensorList(objective.params)
        ndim = params.global_numel()

        grad_step_size = self.defaults['grad']
        adaptive = self.defaults['adaptive']
        index_strategy = self.defaults['index']
        h = self.defaults['h']
        threepoint = self.defaults['threepoint']

        # ------------------------------ determine index ----------------------------- #
        if index_strategy == 'cyclic':
            idx = self.global_state.get('idx', 0) % ndim
            self.global_state['idx'] = idx + 1

        elif index_strategy == 'cyclic2':
            idx = self.global_state.get('idx', 0)
            self.global_state['idx'] = idx + 1
            if idx >= ndim * 2:
                idx = self.global_state['idx'] = 0
            if idx >= ndim:
                idx  = (2*ndim - idx) - 1

        elif index_strategy == 'random':
            if 'generator' not in self.global_state:
                self.global_state['generator'] = random.Random(0)
            generator = self.global_state['generator']
            idx = generator.randrange(0, ndim)

        else:
            raise ValueError(index_strategy)

        # -------------------------- find descent direction -------------------------- #
        h_vec = None
        if adaptive:
            if threepoint:
                h_vec = self.get_state(params, 'h_vec', init=lambda x: torch.full_like(x, h), cls=TensorList)
                h = float(h_vec.flat_get(idx))
            else:
                warnings.warn("CD adaptive=True only works with threepoint=True")

        f_0 = objective.get_loss(False)
        params.flat_set_lambda_(idx, lambda x: x + h)
        f_p = closure(False)

        # -------------------------------- threepoint -------------------------------- #
        if threepoint:
            params.flat_set_lambda_(idx, lambda x: x - 2*h)
            f_n = closure(False)
            params.flat_set_lambda_(idx, lambda x: x + h)

            if adaptive:
                assert h_vec is not None
                if f_0 <= f_p and f_0 <= f_n:
                    h_vec.flat_set_lambda_(idx, lambda x: max(x/2, 1e-10))
                else:
                    if abs(f_0 - f_n) < 1e-12 or abs((f_p - f_0) / (f_0 - f_n) - 1) < 1e-2:
                        h_vec.flat_set_lambda_(idx, lambda x: min(x*2, 1e10))

            if grad_step_size:
                alpha = (f_p - f_n) / (2*h)

            else:
                if f_0 < f_p and f_0 < f_n: alpha = 0
                elif f_p < f_n: alpha = -1
                else: alpha = 1

        # --------------------------------- twopoint --------------------------------- #
        else:
            params.flat_set_lambda_(idx, lambda x: x - h)
            if grad_step_size:
                alpha = (f_p - f_0) / h
            else:
                if f_p < f_0: alpha = -1
                else: alpha = 1

        # ----------------------------- create the update ---------------------------- #
        update = params.zeros_like()
        update.flat_set_(idx, alpha)
        objective.updates = update
        return objective

