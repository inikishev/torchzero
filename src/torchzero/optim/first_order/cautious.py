import typing
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import Cautious, Adam, SGD, LR
from ..modular import Modular


class CautiousAdam(Modular):
    def __init__(
        self,
        params,
        lr: float = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        c_eps = 1e-6,
        normalize = True,
        mode: typing.Literal['zero', 'grad', 'backtrack'] = 'zero'
    ):
        modules: list[OptimizerModule] = [
            Adam(lr = 1 if mode == 'grad' else lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        if mode == 'grad': modules.append(LR(lr))

        super().__init__(params, modules)


class CautiousSGD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = True,
        c_eps = 1e-6,
        normalize = True,
        mode: typing.Literal['zero', 'grad', 'backtrack'] = 'zero'
    ):
        modules: list[OptimizerModule] = [
            SGD(lr = 1 if mode == 'grad' else lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        if mode == 'grad': modules.append(LR(lr))

        super().__init__(params, modules)

