# pylint: disable = singleton-comparison
# ruff: noqa: E712
r"""
This submodule contains composable optimizer "building blocks".
"""
from typing import TypedDict, Any, get_type_hints, Literal

from collections.abc import Iterable, Sequence
from .gradient_approximation import *
from .adaptive import *
from .line_search import *
from .meta import *
from .momentum import *
from .misc import *
from .quasi_newton import *
from .regularization import *
from .second_order import *
from .optimizers import *
from .smoothing import *

from ..core.module import OptimizerModule
# from .experimental.subspace import *

Modules = OptimizerModule | Sequence[OptimizerModule]

def _ismodules(x):
    if isinstance(x, OptimizerModule): return True
    if isinstance(x, Sequence) and len(x)>0 and isinstance(x[0], OptimizerModule): return True
    return False

class _CommonKwargs(TypedDict, total=False):
    # lr: float | Modules | None
    decoupled_l1: float | Modules | None
    """test if this works"""
    decoupled_l2: float | Modules | None
    l1: float | Modules | None
    l2: float | Modules | None
    grad_clip_norm: float | Modules | None
    grad_clip_value: float | Modules | None
    grad_norm: bool | Literal['global', 'param', 'channel'] | Modules | None
    grad_dropout: float | Modules | None
    grad_sign: bool | Modules | None
    update_clip_norm: float | Modules | None
    update_clip_value: float| Modules | None
    update_norm: bool | Modules | None
    update_sign: bool | Modules | None
    lr_dropout: float | Modules | None
    cautious: bool | Modules | None
    line_search: LineSearches | Modules | None
    momentum: float | Modules | None
    dampening: float
    nesterov: bool
    adam: bool | Modules | None
    rmsprop: bool | Modules | None
    laplacian_smoothing: float | Modules | None
    update_laplacian_smoothing: float | Modules | None
    grad_estimator: str | Modules | None
    grad_modules: Modules | None
    update_modules: Modules | None
    main_modules: Modules | None

def _get_module(module:str, arg: Any, all_kwargs: _CommonKwargs):
    skip = {"dampening", "nesterov"}
    if arg in skip: return None
    if arg is None: return None
    if _ismodules(arg): return arg
    if module == 'lr': return LR(arg)
    if module in ("l1", "decoupled_l1"): return WeightDecay(arg, ord = 1)
    if module in ("l2", "decoupled_l2"): return WeightDecay(arg)
    if module in ('grad_clip_norm', 'update_clip_norm'): return ClipNorm(arg)
    if module in ('grad_clip_value', 'update_clip_value'): return ClipValue(arg)
    if module in ('grad_norm', 'update_norm'):
        if arg == True: return Normalize()
        if arg == False: return None
        return Normalize(mode=arg)
    if module in ('grad_dropout', 'lr_dropouts'): return Dropout(arg)
    if module in ('grad_sign', 'update_sign'): return Sign() if arg == True else None
    if module == 'cautious': return Cautious() if arg == True else None
    if module == 'line_search': return get_line_search(arg)
    if module == 'momentum':
        dampening = all_kwargs.get('dampening', 0)
        nesterov = all_kwargs.get('nesterov', False)
        if nesterov: return NesterovMomentum(arg, dampening)
        return HeavyBall(arg, dampening)
    if module == 'nesterov': return NesterovMomentum(arg)
    if module == 'adam': return Adam() if arg == True else None
    if module == 'rmsprop': return RMSProp() if arg == True else None
    if module in ('laplacian_smoothing', 'update_laplacian_smoothing'): return LaplacianSmoothing(arg)
    if module == 'grad_estimator': raise NotImplementedError(module)
    raise ValueError(module)

def _should_decouple_lr(kwargs: _CommonKwargs):
    decoupled_modules = {"update_norm", "update_sign", "update_clip_norm", "update_clip_value"}
    return any(m in kwargs for m in decoupled_modules)

def _get_baked_in_and_module_lr(lr: float, kwargs: _CommonKwargs):
    """some optimizers like adam have `lr` baked in because it is slightly more efficient than using `LR(lr)` module.
    But some modules like update norm require lr to be 1, so an LR(lr) needs to be put after them. Using this basically checks
    if any of those modules are being used and if they are, it sets lr to 1 and appends an LR(lr) module.
    
    .. code:: py
        lr, lr_module = _get_lr_and_lr_module(lr, kwargs)
        main: list[OptimizerModule] = [
            Adam(lr = lr, beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        modules = _make_common_modules(main, lr_module, kwargs)
        super().__init__(params, modules)

    """
    if _should_decouple_lr(kwargs): return 1, lr
    return lr, None

def _make_common_modules(main: OptimizerModule | Iterable[OptimizerModule] | None, lr_module: float | None, kwargs: _CommonKwargs):
    """common modules, this is used to add common things to all torchzero.optim optimizers

    l1 and l2 are applied to the gradient. So in that way they depend on the learning rate of the optimizer.

    Decoupled versions do not depend on learning rate and are applied after the update rule.

    Update modules, such as update_clip_norm, do not depend on lr either.
    """
    from ..python_tools import flatten
    order = [
        "grad_estimator",
        "l1", "l2", "laplacian_smoothing", "grad_modules", "grad_sign", "grad_clip_norm", "grad_clip_value", "grad_norm", "grad_dropout",
        "main", "main_modules", "rmsprop", "adam", "momentum", "cautious", "update_norm",
        "update_laplacian_smoothing", "update_sign", "update_clip_norm", "update_clip_value", "lr",
        "lr_dropout", "decoupled_l1", "decoupled_l2", "update_modules", "line_search"
    ]

    keys = set(get_type_hints(_CommonKwargs).keys()).union({'lr'}).difference({"dampening", "nesterov"})
    order_keys = set(order).difference({'main'})
    assert order_keys == keys, f'missing: {order_keys.difference(keys)}'

    modules_dict = {k: _get_module(k, v, kwargs) for k, v in kwargs.items()}
    modules_dict["main"] = main
    if lr_module is not None: modules_dict['lr'] = _get_module('lr', lr_module, kwargs)

    modules = [modules_dict[k] for k in order if k in modules_dict]
    modules = [i for i in modules if i is not None]
    return flatten(modules)