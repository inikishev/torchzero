from collections.abc import Callable
from functools import partial
from typing import Any, overload

import torch

from ...core import OptimizerModule


class _DummyOpt(torch.optim.Optimizer):
    """dummy optimizer to pass to scheduler"""
    def __init__(self, init_lr:float, init_momentum:float):
        p = torch.tensor(1.)
        p.grad = torch.tensor(1.)
        super().__init__(p, {"lr": init_lr, "momentum": init_momentum})

        self.first_param_group = self.param_groups[0]
        self.step()

    def step(self, closure = None):
        return 1


def _set_momentum_hook(optimizer, state, momentum):
    for module in optimizer.unrolled_modules:
        if 'momentum' in module.defaults:
            for g in module.param_groups:
                g['momentum'] = momentum
        if 'beta1' in module.defaults:
            for g in module.param_groups:
                g['beta1'] = momentum

class LRScheduler(OptimizerModule):
    """Use any pytorch lr scheduler.

    Important - the lr is applied multiplicatively and multiplies with learning rate of other modules,
    so usually base learning rate of the lr scheduler, such as `max_lr` for OneCycleLR, should be set to 1.

    Args:
        lr_scheduler (Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler  |  Any]):
            something like:
            .. code:: py
            lambda opt: OneCycleLR(opt, max_lr = 1, total_steps = 60000)
        update_every (int, optional):
            call `step` every n steps, useful for schedulers that only step once per epoch. Defaults to 1.
        cycle_momentum (bool, optional):
            enables support for cycling momentum with schedulers that support it, such as `OneCycleLR`.
            Unlike lr, momentum is not applied multiplicatively, but set to all other modules with
            `momentum` or `beta` settings. Has no effect if there are no modules that support momentum. Defaults to False.
        init_lr (float, optional):
            initial lr, I believe most lr schedulers ignore this. Defaults to 1.
        init_momentum (float, optional):
            initial init_momentum, I believe most lr schedulers ignore this.
            Has no effect if `cycle_momentum` is False or there are no modules that support momentum. Defaults to 0.
    """
    def __init__(
        self,
        lr_scheduler: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler | Any],
        step_every: int = 1,
        cycle_momentum: bool = False,
        init_lr: float = 1,
        init_momentum: float = 0,
    ):
        super().__init__({})
        self.dummy_opt = _DummyOpt(init_lr, init_momentum)
        scheduler = lr_scheduler(self.dummy_opt)
        self.update_every = step_every
        self.cycle_momentum = cycle_momentum

        self.scheduler_step = scheduler.step
        self.cur = 0
        self.cur_lr = init_lr
        self.cur_momentum = init_momentum

    def step(self, state):
        if self.cur % self.update_every == 0:
            self.scheduler_step()
            self.cur_lr = self.dummy_opt.first_param_group['lr']
            self.cur_momentum = self.dummy_opt.first_param_group['momentum']

        params = self.get_params()
        ascent = state.maybe_use_grad_(params)
        ascent *= self.cur_lr

        if self.cycle_momentum:
            state.add_post_step_hook(partial(_set_momentum_hook, momentum = self.cur_momentum))

class LRWarmup(OptimizerModule):
    """Linear learning rate warmup.

    Args:
        n_steps (int): number of warmup steps.
        start_lr (float, optional): initial lr. Defaults to 1e-8.
        end_lr (float, optional): final lr. Defaults to 1.
        delay_steps (int, optional): number of `start_lr` steps before starting the warmup. Defaults to 0.
    """
    def __init__(self, n_steps: int, start_lr: float = 1e-8, end_lr: float = 1, delay_steps: int = 0):

        super().__init__({})
        self.n_steps = n_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.delay_steps = delay_steps

        self.cur = 0

    def _update(self, state, ascent):
        if self.cur < self.delay_steps:
            if self.start_lr != 1: ascent *= self.start_lr

        elif self.cur >= self.n_steps + self.delay_steps:
            if self.end_lr != 1: ascent *= self.end_lr

        else:
            remaining = (self.n_steps - (self.cur-self.delay_steps)) / self.n_steps
            lr = (self.start_lr * remaining) + self.end_lr * (1 - remaining)
            ascent *= lr

        self.cur += 1
        return ascent


