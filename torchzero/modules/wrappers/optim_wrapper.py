from collections.abc import Iterable, Mapping, Sequence, Callable
from typing import Any
import torch

from ...core.module import Module
from ...utils.params import Params, _copy_param_groups, _make_param_groups


class Wrap(Module):
    """
    Wraps a pytorch optimizer to use it as a module.

    Note:
        Custom param groups are supported only by ``set_param_groups``, settings passed to Optimizer will be applied to all parameters.

    Args:
        opt_fn (Callable[..., torch.optim.Optimizer] | torch.optim.Optimizer):
            function that takes in parameters and returns the optimizer, for example ``torch.optim.Adam``
            or ``lambda parameters: torch.optim.Adam(parameters, lr=1e-3)``
        *args:
        **kwargs:
            Extra args to be passed to opt_fn. The function is called as ``opt_fn(parameters, *args, **kwargs)``.
        use_param_groups:
            Whether to pass settings passed to Optimizer to the wrapped optimizer.

            Note that settings to the first parameter are used for all parameters,
            so if you specified per-parameter settings, they will be ignored.

    ### Example:
    wrapping pytorch_optimizer.StableAdamW

    ```python

    from pytorch_optimizer import StableAdamW
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Wrap(StableAdamW, lr=1),
        tz.m.Cautious(),
        tz.m.LR(1e-2)
    )
    ```

    """

    def __init__(
        self,
        opt_fn: Callable[..., torch.optim.Optimizer] | torch.optim.Optimizer,
        *args,
        use_param_groups: bool = True,
        **kwargs,
    ):
        defaults = dict(use_param_groups=use_param_groups)
        super().__init__(defaults=defaults)

        self._opt_fn = opt_fn
        self._opt_args = args
        self._opt_kwargs = kwargs
        self._custom_param_groups = None

        self.optimizer: torch.optim.Optimizer | None = None
        if isinstance(self._opt_fn, torch.optim.Optimizer) or not callable(self._opt_fn):
            self.optimizer = self._opt_fn

    def set_param_groups(self, param_groups):
        self._custom_param_groups = _make_param_groups(param_groups, differentiable=False)
        return super().set_param_groups(param_groups)

    @torch.no_grad
    def apply(self, objective):
        params = objective.params

        # initialize opt on 1st step
        if self.optimizer is None:
            assert callable(self._opt_fn)
            param_groups = params if self._custom_param_groups is None else self._custom_param_groups
            self.optimizer = self._opt_fn(param_groups, *self._opt_args, **self._opt_kwargs)

        # set optimizer per-parameter settings
        if self.defaults["use_param_groups"] and objective.modular is not None:
            for group in self.optimizer.param_groups:
                first_param = group['params'][0]
                setting = self.settings[first_param]

                # settings passed in `set_param_groups` are the highest priority
                # schedulers will override defaults but not settings passed in `set_param_groups`
                # this is consistent with how Optimizer does it.
                if self._custom_param_groups is not None:
                    setting = {k:v for k,v in setting if k not in self._custom_param_groups[0]}

                group.update(setting)

        # set grad to update
        orig_grad = [p.grad for p in params]
        for p, u in zip(params, objective.get_updates()):
            p.grad = u

        # if this is last module, simply use optimizer to update parameters
        if objective.modular is not None and self is objective.modular.modules[-1]:
            self.optimizer.step()

            # restore grad
            for p, g in zip(params, orig_grad):
                p.grad = g

            objective.stop = True; objective.skip_update = True
            return objective

        # this is not the last module, meaning update is difference in parameters
        # and passed to next module
        params_before_step = [p.clone() for p in params]
        self.optimizer.step() # step and update params
        for p, g in zip(params, orig_grad):
            p.grad = g
        objective.updates = list(torch._foreach_sub(params_before_step, params)) # set update to difference between params
        for p, o in zip(params, params_before_step):
            p.set_(o) # pyright: ignore[reportArgumentType]

        return objective

    def reset(self):
        super().reset()
        assert self.optimizer is not None
        for g in self.optimizer.param_groups:
            for p in g['params']:
                state = self.optimizer.state[p]
                state.clear()
