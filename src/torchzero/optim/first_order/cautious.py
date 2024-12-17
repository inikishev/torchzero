import typing
from collections import abc

import torch

from ...core import OptimizerModule
from ...modules import Cautious, Adam, SGD, LR
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
        c_eps = 1e-6,
        normalize = True,
        mode: typing.Literal['zero', 'grad', 'negate'] = 'zero'
    ):
        """Cautious adam.

        Args:
            params (_type_): iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1.
            beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
            beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
            eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
            amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm from the paper
                On the Convergence of Adam and Beyond (default: False).
        """
        modules: list[OptimizerModule] = [
            Adam(lr = 1 if mode == 'grad' else lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        if mode == 'grad': modules.append(LR(lr))

        super().__init__(params, modules)


class CautiousSGD(ModularOptimizer):
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
        mode: typing.Literal['zero', 'grad', 'negate'] = 'zero'
    ):
        """Cautious SGD with momentum (without momentum this is just SGD)

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 1e-3.
            momentum (float, optional): _description_. Defaults to 0.99.
            dampening (float, optional): _description_. Defaults to 0.
            weight_decay (float, optional): _description_. Defaults to 0.
            nesterov (bool, optional): _description_. Defaults to True.
        """
        modules: list[OptimizerModule] = [
            SGD(lr = 1 if mode == 'grad' else lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        if mode == 'grad': modules.append(LR(lr))

        super().__init__(params, modules)

