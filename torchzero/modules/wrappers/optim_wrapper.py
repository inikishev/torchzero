from collections.abc import Iterable, Mapping, Sequence, Callable
from typing import Any
import torch

from ...core.module import Module
from ...utils import Params, _copy_param_groups, _make_param_groups


class Wrap(Module):
    """
    Wraps a pytorch optimizer to use it as a module.

    .. note::
        Custom param groups are supported only by `set_param_groups`, settings passed to Modular will be ignored.

    Args:
        opt_fn (Callable[..., torch.optim.Optimizer] | torch.optim.Optimizer):
            function that takes in parameters and returns the optimizer, for example :code:`torch.optim.Adam`
            or :code:`lambda parameters: torch.optim.Adam(parameters, lr=1e-3)`
        *args:
        **kwargs:
            Extra args to be passed to opt_fn. The function is called as :code:`opt_fn(parameters, *args, **kwargs)`.

    Example:
        wrapping pytorch_optimizer.StableAdamW

        .. code-block:: py

            from pytorch_optimizer import StableAdamW
            opt = tz.Modular(
                model.parameters(),
                tz.m.Wrap(StableAdamW, lr=1),
                tz.m.Cautious(),
                tz.m.LR(1e-2)
            )


    """
    def __init__(self, opt_fn: Callable[..., torch.optim.Optimizer] | torch.optim.Optimizer, *args, **kwargs):
        super().__init__()
        self._opt_fn = opt_fn
        self._opt_args = args
        self._opt_kwargs = kwargs
        self._custom_param_groups = None

        self.optimizer: torch.optim.Optimizer | None = None
        if isinstance(self._opt_fn, torch.optim.Optimizer) or not callable(self._opt_fn):
            self.optimizer = self._opt_fn

    def set_param_groups(self, param_groups):
        self._custom_param_groups = param_groups
        return super().set_param_groups(param_groups)

    @torch.no_grad
    def step(self, var):
        params = var.params

        # initialize opt on 1st step
        if self.optimizer is None:
            assert callable(self._opt_fn)
            param_groups = params if self._custom_param_groups is None else self._custom_param_groups
            self.optimizer = self._opt_fn(param_groups, *self._opt_args, **self._opt_kwargs)

        # set grad to update
        orig_grad = [p.grad for p in params]
        for p, u in zip(params, var.get_update()):
            p.grad = u

        # if this is last module, simply use optimizer to update parameters
        if var.modular is not None and self is var.modular.modules[-1]:
            self.optimizer.step()

            # restore grad
            for p, g in zip(params, orig_grad):
                p.grad = g

            var.stop = True; var.skip_update = True
            return var

        # this is not the last module, meaning update is difference in parameters
        # and passed to next module
        params_before_step = [p.clone() for p in params]
        self.optimizer.step() # step and update params
        for p, g in zip(params, orig_grad):
            p.grad = g
        var.update = list(torch._foreach_sub(params_before_step, params)) # set update to difference between params
        for p, o in zip(params, params_before_step):
            p.set_(o) # pyright: ignore[reportArgumentType]

        return var

    def reset(self):
        super().reset()
        assert self.optimizer is not None
        for g in self.optimizer.param_groups:
            for p in g['params']:
                state = self.optimizer.state[p]
                state.clear()
