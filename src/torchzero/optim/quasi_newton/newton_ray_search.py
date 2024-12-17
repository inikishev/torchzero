import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import (SGD, HvInvFDM, LineSearches, NewtonFDM, Subspace,
                        get_line_search, ProjAscentRay, ProjNormalize, LR, OptimizerWrapper)
from ..modular import ModularOptimizer


class NewtonFDMRaySearch(ModularOptimizer):
    def __init__(
        self,
        params,
        lr = 1e-2,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        n_rays = 3,
        eps = 1e-2,
        ray_width: float = 1e-1,
        line_search: LineSearches | None = 'brent'
    ):
        """experimental (probably failed) please don't use"""
        modules: list[OptimizerModule] = [
            SGD(1, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov),
            Subspace(NewtonFDM(eps = eps), ProjNormalize(ProjAscentRay(ray_width, n = n_rays))),
        ]
        if lr != 1:
            modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)


class LBFGSRaySearch(ModularOptimizer):
    def __init__(
        self,
        params,
        lr = 1,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        n_rays = 24,
        ray_width: float = 1e-1,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | T.Literal['strong_wolfe'] | None = None,
    ):
        """experimental (probably failed) please don't use"""
        lbfgs = OptimizerWrapper(
                torch.optim.LBFGS,
                pass_closure = True,
                lr = lr,
                max_iter = max_iter,
                max_eval = max_eval,
                tolerance_grad = tolerance_grad,
                tolerance_change = tolerance_change,
                history_size = history_size,
                line_search_fn = line_search_fn,
            )
        modules: list[OptimizerModule] = [
            SGD(1, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov),
            Subspace(lbfgs, ProjNormalize(ProjAscentRay(ray_width, n = n_rays))),

        ]

        super().__init__(params, modules)



