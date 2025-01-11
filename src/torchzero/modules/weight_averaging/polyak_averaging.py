from collections.abc import Callable, Iterable
from warnings import warn

import torch

from torchzero.tensorlist import TensorList

from ...core import OptimizerModule


class PolyakAveraging(OptimizerModule):
    """Every n steps this sets parameters to the average over last n steps.

    Please put this module at the end, after all other modules.

    Args:
        n_steps (int): number of steps (batches).
    """
    def __init__(self, n_steps: int):

        super().__init__({})
        self.n_steps = n_steps
        self.cur_step = 0

    @torch.no_grad
    def step(self, state):
        if self.next_module is not None:
            warn(f'PolyakAveraging should usually be the last module, but {self.next_module.__class__.__name__} is after it.')
        self.cur_step += 1

        params = self.get_params()
        # update params with the child. Averaging is always applied at the end.
        state.maybe_use_grad_(params)
        self._update_params_or_step_with_next(state, params)

        sum = self.get_state_key('sum')
        sum += params

        if self.cur_step % self.n_steps == 0:
            params.set_(sum / self.n_steps)
            sum.zero_()

        return state.get_loss()


class EMA(OptimizerModule):
    """Uses exponential moving average of past weights.

    Please put this module at the end, after all other modules.

    Args:
        n_steps (int): number of steps (batches).
    """
    def __init__(self, momentum=0.9999, update_every = 1):
        defaults = dict(momentum=momentum)
        super().__init__(defaults)
        self.update_every = update_every
        self.cur_step = 0
        self.update_every = update_every

    @torch.no_grad
    def step(self, state):
        if self.next_module is not None:
            warn(f'EMA should usually be the last module, but {self.next_module.__class__.__name__} is after it.')
        self.cur_step += 1

        params = self.get_params()
        # update params with the child. Averaging is always applied at the end.
        state.maybe_use_grad_(params)
        self._update_params_or_step_with_next(state, params)

        ema = self.get_state_key('ema', init = 'params', params=params)
        momentum = self.get_group_key('momentum')

        ema.lerp_compat_(params, 1 - momentum)

        if self.cur_step % self.update_every == 0:
            params.set_(ema.clone())

        return state.get_loss()