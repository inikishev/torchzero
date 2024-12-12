import typing as T
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import Cautious, Adam, SGD
from ..modular import ModularOptimizer


class CautiousAdam(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
    ):
        """Cautious adam.

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 1.
            beta1 (float, optional): _description_. Defaults to 0.9.
            beta2 (float, optional): _description_. Defaults to 0.999.
            eps (float, optional): _description_. Defaults to 1e-8.
            amsgrad (bool, optional): _description_. Defaults to False.
        """
        modules: list[OptimizerModule] = [
            Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            Cautious(),
        ]

        super().__init__(params, modules)


class CautiousSGD(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.99,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = True,
    ):
        """Cautious SGD with momentum (without momentum this does nothing)

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 1e-3.
            momentum (float, optional): _description_. Defaults to 0.99.
            dampening (float, optional): _description_. Defaults to 0.
            weight_decay (float, optional): _description_. Defaults to 0.
            nesterov (bool, optional): _description_. Defaults to True.
        """
        modules: list[OptimizerModule] = [
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
            Cautious(),
        ]

        super().__init__(params, modules)

