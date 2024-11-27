import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import (SGD, NewtonGradFDM, LineSearches, NewtonFDM, Subspace,
                        get_line_search, ProjAscentRay)
from ..modular import ModularOptimizer


class NewtonRaySearch(ModularOptimizer):
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
        modules: list[OptimizerModule] = [
            SGD(lr = lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov),
            Subspace(ProjAscentRay(ray_width, n = n_rays)),
            NewtonFDM(eps = eps),
        ]

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)

class DiagNewtonRaySearch(ModularOptimizer):
    def __init__(
        self,
        params,
        lr = 1e-2,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        n_rays = 24,
        eps = 1e-2,
        ray_width: float = 1e-1,
        line_search: LineSearches | None = None,
    ):

        modules: list[OptimizerModule] = [
            SGD(lr = lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov),
            Subspace(ProjAscentRay(ray_width, n = n_rays)),
            NewtonGradFDM(eps = eps),
        ]

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)


