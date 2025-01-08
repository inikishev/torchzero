from typing import Literal, Unpack

from ...core import OptimizerModule
from ...modules import Cautious, Adam, SGD, Lion, _make_common_modules, _CommonKwargs, _get_baked_in_and_module_lr
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
        normalize = False,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)

        main: list[OptimizerModule] = [
            Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        modules = _make_common_modules(main, lr_module, kwargs)
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
        normalize = False,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)

        main: list[OptimizerModule] = [
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]

        modules = _make_common_modules(main, lr_module, kwargs)

        super().__init__(params, modules)


class CautiousLion(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        c_eps = 1e-6,
        normalize = False,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        main: list[OptimizerModule] = [
            Lion(beta1, beta2),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]

        modules = _make_common_modules(main, lr, kwargs)

        super().__init__(params, modules)
