import typing
from collections import abc

import numpy as np
import torch

import nevergrad as ng

from .wrapper import WrapperBase


def _ensure_float(x) -> float:
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return float(x.item())
    return float(x)

class NevergradWrapper(WrapperBase):
    """Use nevergrad optimizer as pytorch optimizer.
    Note that it is recommended to specify `budget` to the number of iterations you expect to run,
    as some nevergrad optimizers will error without it.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        opt_cls (type[ng.optimizers.base.Optimizer]):
            nevergrad optimizer class. For example, `ng.optimizers.NGOpt`.
        budget (int | None, optional):
            nevergrad parameter which sets allowed number of function evaluations (forward passes).
            This only affects the behaviour of many nevergrad optimizers, for example some
            use certain rule for first 50% of the steps, and then switch to another rule.
            This parameter doesn't actually limit the maximum number of steps!
            But it doesn't have to be exact. Defaults to None.
        lb (float | None, optional):
            lower bounds, this can also be specified in param_groups. Bounds are optional, however
            some nevergrad algorithms will raise an exception of bounds are not specified.
        ub (float, optional):
            upper bounds, this can also be specified in param_groups. Bounds are optional, however
            some nevergrad algorithms will raise an exception of bounds are not specified.
        mutable_sigma (bool, optional):
            nevergrad parameter, sets whether the mutation standard deviation must mutate as well
            (for mutation based algorithms). Defaults to False.
        use_init (bool, optional):
            whether to use initial model parameters as initial parameters for the nevergrad parametrization.
            The reason you might want to set this to False is because True seems to break some optimizers
            (mainly portfolio ones by initalizing them all to same parameters so they all perform exactly the same steps).
            However if you are fine-tuning something, you have to set this to True, otherwise it will start from
            new random parameters. Defaults to True.
    """
    def __init__(
        self,
        params,
        opt_cls:"type[ng.optimizers.base.Optimizer] | abc.Callable[..., ng.optimizers.base.Optimizer]",
        budget: int | None = None,
        lb: float | None = None,
        ub: float | None = None,
        mutable_sigma = False,
        use_init = True,
    ):
        defaults = dict(lb=lb, ub=ub, use_init=use_init, mutable_sigma=mutable_sigma)
        super().__init__(params, defaults)
        self.opt_cls = opt_cls
        self.opt = None
        self.budget = budget

    @torch.no_grad
    def step(self, closure): # pylint:disable=signature-differs # pyright:ignore[reportIncompatibleMethodOverride]
        params = self._get_params()
        if self.opt is None:
            ng_params = []
            for group in self.param_groups:
                params = group['params']
                mutable_sigma = group['mutable_sigma']
                use_init = group['use_init']
                lb = group['lb']
                ub = group['ub']
                for p in params:
                    if p.requires_grad:
                        if use_init:
                            ng_params.append(
                                ng.p.Array(init = p.detach().cpu().numpy(), lower=lb, upper=ub, mutable_sigma=mutable_sigma))
                        else:
                            ng_params.append(
                                ng.p.Array(shape = p.shape, lower=lb, upper=ub, mutable_sigma=mutable_sigma))

            parametrization = ng.p.Tuple(*ng_params)
            self.opt = self.opt_cls(parametrization, budget=self.budget)

        x: ng.p.Tuple = self.opt.ask() # type:ignore
        for cur, new in zip(params, x):
            cur.set_(torch.as_tensor(new.value, dtype=cur.dtype, device=cur.device).reshape_as(cur)) # type:ignore

        loss = closure(False)
        self.opt.tell(x, _ensure_float(loss))
        return loss
