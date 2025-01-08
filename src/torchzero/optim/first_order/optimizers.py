from typing import Unpack, Literal

from ...modules import SGD as _SGD
from ...modules import Adagrad as _Adagrad
from ...modules import Adam as _Adam
from ...modules import LineSearches, Normalize, Centralize
from ...modules import RMSProp as _RMSProp
from ...modules import Rprop as _Rprop
from ...modules import Lion as _Lion
from ...modules import UseGradSign
from ...modules import (Sign, _CommonKwargs, _get_baked_in_and_module_lr,
                        _make_common_modules)
from ..modular import Modular


class GD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1,
        line_search: LineSearches | None = 'backtracking',
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        kwargs['line_search'] = line_search
        modules = _make_common_modules(None, lr, kwargs)
        super().__init__(params, modules)

class SGD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = weight_decay, nesterov = nesterov)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class SignSGD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        modules = _make_common_modules(Sign(), lr, kwargs)
        super().__init__(params, modules)


class NormSGD(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        centralize=False,
        norm_mode: Literal["global", "param", "channel"] = 'param',
        centralize_mode: Literal["global", "param", "channel"] = 'channel',
        min_numel=2,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        
        main: list = [Normalize(lr, mode=norm_mode, min_numel=min_numel)]
        if centralize: main.append(Centralize(centralize_mode, min_numel=min_numel))
        
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)


class Adagrad(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3, lr_decay: float = 0, initial_accumulator_value: float = 0, eps: float = 1e-10,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adagrad(lr = lr, lr_decay = lr_decay, initial_accumulator_value = initial_accumulator_value, eps = eps)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class Rprop(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Rprop(lr = lr, nplus = nplus, nminus = nminus, lb=lb, ub = ub, backtrack=backtrack)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class RMSProp(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8, centered: bool = False,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        main = _RMSProp(alpha = alpha, eps = eps, centered = centered,)
        modules = _make_common_modules(main, lr, kwargs)
        super().__init__(params, modules)

class Adam(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

class AdamW(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        """Adam with decoupled weight decay. Weight decay doesn't depend on learning rate and is applied after adam update rule.

        Args:
            params (_type_): _description_
            lr (float, optional): _description_. Defaults to 1.
            beta1 (float, optional): _description_. Defaults to 0.9.
            beta2 (float, optional): _description_. Defaults to 0.999.
            eps (float, optional): _description_. Defaults to 1e-8.
            weight_decay (float, optional): _description_. Defaults to 0.01.
            amsgrad (bool, optional): _description_. Defaults to False.
        """
        kwargs['decoupled_l2'] = weight_decay
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad)
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)
        

class Grams(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        """adam but uses gradient sign https://arxiv.org/abs/2412.17107v1"""
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            _Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            UseGradSign()
        ]
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)
        
        
class Lion(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        modules = _make_common_modules(_Lion(beta1, beta2), lr, kwargs)
        super().__init__(params, modules)