from collections.abc import Callable, Iterable

import torch

from ...tensorlist import TensorList

from ...core import OptimizerModule


class Accumulate(OptimizerModule):
    """Accumulates update over n steps, and steps once updates have been accumulated.
    Put this as the first module to get gradient accumulation.

    Args:
        n_steps (int): number of steps (batches) to accumulate the update over.
        mean (bool, optional):
            If True, divides accumulated gradients by number of step,
            since most loss functions calculate the mean of all samples
            over batch. Defaults to True.
    """
    def __init__(self, n_steps: int, mean = True):

        super().__init__({})
        self.n_steps = n_steps
        self.mean = mean
        self.cur_step = 0

    @torch.no_grad
    def step(self, vars):
        self.cur_step += 1

        params = self.get_params()
        accumulated_update = self.get_state_key('accumulated_grads')
        accumulated_update += vars.maybe_use_grad_(params)

        if self.cur_step % self.n_steps == 0:
            vars.ascent = accumulated_update.clone()
            if self.mean: vars.ascent /= self.n_steps
            accumulated_update.zero_()
            return self._update_params_or_step_with_next(vars)


        return vars.get_loss()