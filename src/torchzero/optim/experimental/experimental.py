from collections.abc import Iterable
from typing import Literal, Unpack

from ...modules import (
    SGD,
    Abs,
    Adam,
    Add,
    AddMagnitude,
    Cautious,
    Div,
    Divide,
    Grad,
    HeavyBall,
    Interpolate,
    Lerp,
    NanToNum,
    NesterovMomentum,
    Normalize,
    RDiv,
    Reciprocal,
    ReduceOutwardLR,
    _CommonKwargs,
    _get_baked_in_and_module_lr,
    _make_common_modules,
)
from ...modules import RandomCoordinateMomentum as _RandomCoordinateMomentum
from ...modules.experimental.gradmin import GradMin as _GradMin
from ...modules.experimental.squared_grad_norm_fdm import (
    SquaredGradientNormFDM as _SquaredGradientNormFDM,
)
from ..modular import Modular


class SquaredGradientNormFDM(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this should approximate newton method with 2 backward passes, but only if hessian is purely diagonal"""
    def __init__(
        self,
        params,
        lr: float = 1,
        eps: float = 1e-2,
        **kwargs: Unpack[_CommonKwargs]
    ):
        modules = _make_common_modules(_SquaredGradientNormFDM(eps = eps), lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)


class ReciprocalSGD(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this basically uses normalized *1 / (gradient + eps)*."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-2,
        **kwargs: Unpack[_CommonKwargs]
    ):

        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            AddMagnitude(eps, add_to_zero=False),
            Reciprocal(),
            NanToNum(0,0,0),
            Normalize(lr)
        ]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)


class MomentumNumerator(Modular):
    """for experiments, unlikely to work well on most problems. (this one is promising)

    explanation - momentum divided by gradient."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        eps: float = 1e-2,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):

        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            Divide(
                numerator = SGD(1, momentum, nesterov=nesterov),
                denominator=[Abs(), Add(eps)]
            ),
            Normalize(lr)
        ]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)

class MomentumDenominator(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - gradient divided by normalized momentum."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        eps: float = 1e-2,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):

        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            Div([SGD(1, momentum, nesterov=nesterov), Abs(), Add(eps), Normalize(1)]),
            Normalize(lr)
        ]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)


class ExaggeratedNesterov(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - exaggerates difference between heavyball and nesterov momentum."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        dampening: float = 0,
        strength: float = 5,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):

        main = [
            Interpolate(HeavyBall(momentum, dampening), NesterovMomentum(momentum, dampening), strength),
        ]
        modules = _make_common_modules(main, lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)

class ExtraCautiousAdam(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - exaggerates caution."""
    def __init__(
        self,
        params,
        lr: float = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad=False,
        normalize = False,
        c_eps = 1e-6,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        strength = 5,
        **kwargs: Unpack[_CommonKwargs],
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [
            Adam(lr, beta1, beta2, eps, amsgrad),
            Lerp(Cautious(normalize, c_eps, mode), strength),
        ]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)

class InwardSGD(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - reduces lrs for updates that move weights away from 0."""
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        mul = 0.5,
        use_grad=False,
        **kwargs: Unpack[_CommonKwargs], # type:ignore
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)

        main = [
            SGD(lr, momentum, dampening, weight_decay, nesterov),
            ReduceOutwardLR(mul, use_grad)
        ]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)



class RandomCoordinateMomentum(Modular):
    """for experiments, unlikely to work well on most problems.

    Only uses `p` random coordinates of the new update. Other coordinates remain from previous update.
    This works but I don't know if it is any good.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        p (float, optional): probability to update velocity with a new weigh value. Defaults to 0.1.
        nesterov (bool, optional): if False, update uses delayed momentum. Defaults to True.

    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        p: float = 0.1,
        nesterov: bool = True,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):
        main = _RandomCoordinateMomentum(p, nesterov)
        modules = _make_common_modules(main, lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)

class GradMin(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this uses gradient wrt sum of gradients + loss."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        add_loss: float = 1,
        square: bool = False,
        maximize_grad: bool = False,
        **kwargs: Unpack[_CommonKwargs],
    ):
        main = _GradMin(add_loss, square, maximize_grad)
        modules = _make_common_modules(main, lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)


