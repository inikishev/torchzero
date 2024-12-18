import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import SGD, Sign, Adam
from ..modular import Modular


class SignSGD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = True,
    ):
        modules: list[OptimizerModule] = [
            Sign(),
            SGD(lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
        ]

        super().__init__(params, modules)

