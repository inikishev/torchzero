import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import SGD, Sign, Adam
from ..modular import ModularOptimizer


class SignSGD(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = True,
    ):
        """Sign SGD (can be better than SGD).

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 1e-3.
            momentum (float, optional): _description_. Defaults to 0.99.
            dampening (float, optional): _description_. Defaults to 0.
            weight_decay (float, optional): _description_. Defaults to 0.
            nesterov (bool, optional): _description_. Defaults to True.
        """
        modules: list[OptimizerModule] = [
            Sign(),
            SGD(lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
        ]

        super().__init__(params, modules)

