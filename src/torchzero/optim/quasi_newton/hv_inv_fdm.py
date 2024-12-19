import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules.quasi_newton.hv_inv_fdm import HvInvFDM as _HvInvFDM
from ...modules import get_line_search, LineSearches, LR
from ..modular import Modular


class HvInvFDM(Modular):
    """Experimental (maybe don't use yet)."""
    def __init__(
        self,
        params,
        lr: float = 1,
        eps: float = 1e-2,
        line_search: LineSearches | None = None,
    ):
        modules: list[OptimizerModule] = [
            _HvInvFDM(eps = eps),
        ]

        if lr != 1:
            modules.append(LR(lr))

        if line_search is not None:
            modules.append(get_line_search(line_search))

        super().__init__(params, modules)

