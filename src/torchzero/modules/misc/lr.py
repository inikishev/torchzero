import random
from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, overload

import torch

from ...tensorlist import TensorList

from ...core import OptimizerModule

if TYPE_CHECKING:
    from ...optim import Modular

def _init_scheduler_hook(opt: "Modular", module: "LR", scheduler_cls, **kwargs):
    """post init hook that initializes the lr scheduler to the LR module and sets `_scheduler_step_fn`."""
    scheduler = scheduler_cls(module, **kwargs)
    module._scheduler_step_fn = scheduler.step

def _set_momentum_hook(optimizer, state, momentum):
    for module in optimizer.unrolled_modules:
        if 'momentum' in module.defaults:
            for g in module.param_groups:
                g['momentum'] = momentum
        elif 'beta1' in module.defaults:
            for g in module.param_groups:
                g['beta1'] = momentum

class LR(OptimizerModule):
    """Multiplies update by the learning rate. Optionally uses an lr scheduler.

    Args:
        lr (float, optional): learning rate. Defaults to 1e-3.
        scheduler (Callable[..., torch.optim.lr_scheduler.LRScheduler  |  Any] | None, optional):
            A scheduler class, for example `torch.optim.lr_scheduler.OneCycleLR`. Defaults to None.
        cycle_momentum (bool, optional):
            enables schedulers that support it to affect momentum (like OneCycleLR).
            The momentum will be cycled on ALL modules that have `momentum` or `beta1` setting.
            This does not support external optimizers, wrapped with `Wrap`. Defaults to True.
        sheduler_step_every (int, optional):
            step with scheduler every n optimizer steps.
            Useful when the scheduler steps once per epoch. Defaults to 1.
        **kwargs:
            kwargs to pass to `scheduler`.
    """
    IS_LR_MODULE = True
    def __init__(
        self,
        lr: float = 1e-3,
        scheduler_cls: Callable[..., torch.optim.lr_scheduler.LRScheduler | Any] | None = None,
        cycle_momentum: bool = True,
        sheduler_step_every: int = 1,
        # *args,
        **kwargs,
    ):

        defaults = dict(lr = lr)

        if (scheduler_cls is not None) and cycle_momentum:
            defaults['momentum'] = 0
        super().__init__(defaults)

        self._scheduler_step_fn = None
        self.sheduler_step_every = sheduler_step_every
        self.cycle_momentum = cycle_momentum
        self.cur = 0

        if scheduler_cls is not None:
            self.post_init_hooks.append(lambda opt, module: _init_scheduler_hook(opt, module, scheduler_cls, **kwargs))

        self._skip = False

    @torch.no_grad
    def _update(self, state, ascent):
        # step with scheduler
        if self._scheduler_step_fn is not None:
            if self.cur != 0 and self.cur % self.sheduler_step_every == 0:
                self._scheduler_step_fn()

                # add a hook to cycle momentum
                if self.cycle_momentum:
                    state.add_post_step_hook(_set_momentum_hook)

            # remove init hook to delete reference to scheduler
            if self.cur == 0 and len(self.post_init_hooks) == 1:
                del self.post_init_hooks[0]

        # skip if lr was applied by previous module (LR fusing)
        if not self._skip:
            # multiply ascent direction by lr in-place
            lr = self.get_group_key('lr')
            ascent *= lr

        self.cur += 1
        self._skip = False
        return ascent