import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import HvInvFDM as _HvInvFDM, get_line_search, LineSearches, LR
from ..modular import ModularOptimizer


class HvInvFDM(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1,
        eps: float = 1e-2,
        line_search: LineSearches | None = None,
    ):
        """Experimental (maybe don't use yet).

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            eps (float, optional): epsilon for finite difference.
                Note that with float32 this needs to be quite high to avoid numerical instability. Defaults to 1e-2.
            line_search (OptimizerModule | None, optional): line search module, can be None. Defaults to None.
        """
        modules: list[OptimizerModule] = [
            _HvInvFDM(eps = eps),
        ]

        if lr != 1:
            modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)

