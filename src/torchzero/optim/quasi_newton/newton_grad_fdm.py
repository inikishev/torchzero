import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import NewtonGradFDM as _NewtonGradFDM, get_line_search, LineSearches, LR
from ..modular import ModularOptimizer


class NewtonGradFDM(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1,
        eps: float = 1e-2,
        line_search: LineSearches | None = 'brent'
    ):
        """Newton approximated in two forwards and backwards from two gradients.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            eps (float, optional): epsilon for finite difference.
                Note that with float32 this needs to be quite high to avoid numerical instability. Defaults to 1e-2.
            line_search (OptimizerModule | None, optional): line search module, can be None. Defaults to "brent".
        """
        modules: list[OptimizerModule] = [
            _NewtonGradFDM(eps = eps),
        ]

        if lr != 1:
            modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)

