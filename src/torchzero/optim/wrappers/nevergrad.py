import typing
from collections import abc

import nevergrad as ng
import numpy as np
import torch

from ...core import ClosureType, TensorListOptimizer
from ...modules import (Proj2Masks, ProjGrad, ProjLastGradDifference,
                        ProjNormalize, Subspace,
                        UninitializedClosureOptimizerWrapper)
from ..modular import ModularOptimizer


def _ensure_float(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().item()
    elif isinstance(x, np.ndarray): return x.item()
    return float(x)

class NevergradOptimizer(TensorListOptimizer):
    def __init__(
        self,
        params,
        opt_cls:"type[ng.optimizers.base.Optimizer] | abc.Callable[..., ng.optimizers.base.Optimizer]",
        budget=None,
        mutable_sigma = False,
        use_init = True,
    ):
        """Use nevergrad optimizer as pytorch optimizer.
        Note that it is recommended to specify `budget`, in fact some optimizers won't work without it.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            opt_cls (type[ng.optimizers.base.Optimizer]):
                nevergrad optimizer class. For example, `ng.optimizers.NGOpt`.
            budget (_type_, optional):
                nevergrad parameter which sets allowed number of function evaluations (forward passes).
                This only affects the behaviour of many nevergrad optimizers, for example some
                use certain rule for first 50% of the steps, and then switch to another rule.
                This parameter doesn't actually limit the maximum number of steps!
                But it doesn't have to be exact. Defaults to None.
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
        super().__init__(params, {})
        self.opt_cls = opt_cls
        self.opt = None
        self.budget = budget
        self.mutable_sigma = mutable_sigma
        self.use_init = use_init

    @torch.no_grad
    def step(self, closure): # type:ignore # pylint:disable=signature-differs
        params = self.get_params()
        if self.opt is None:

            if self.use_init:
                parametrization = ng.p.Tuple(*(ng.p.Array(init = p.detach().cpu().numpy(), mutable_sigma=self.mutable_sigma) for p in params))
            else:
                parametrization = ng.p.Tuple(*(ng.p.Array(shape = p.shape, mutable_sigma=self.mutable_sigma) for p in params))

            self.opt = self.opt_cls(parametrization, budget=self.budget)

        x: ng.p.Tuple = self.opt.ask() # type:ignore
        for cur, new in zip(params, x):
            cur.set_(torch.from_numpy(new.value).to(dtype=cur.dtype, device=cur.device, copy=False).reshape_as(cur)) # type:ignore

        loss = closure(False)
        self.opt.tell(x, _ensure_float(loss))
        return loss



# class NevergradSubspace(ModularOptimizer):
#     def __init__(
#         self,
#         params,
#         opt_cls:"type[ng.optimizers.base.Optimizer] | abc.Callable[..., ng.optimizers.base.Optimizer]",
#         budget=None,
#         mutable_sigma = False,
#         use_init = True,
#         projections = Proj2Masks(5),
#     ):

#         modules = [
#             Subspace(projections, update_every=100),
#             UninitializedClosureOptimizerWrapper(
#                 NevergradOptimizer,
#                 opt_cls = opt_cls,
#                 budget = budget,
#                 mutable_sigma = mutable_sigma,
#                 use_init = use_init,
#             ),
#         ]

#         super().__init__(params, modules)