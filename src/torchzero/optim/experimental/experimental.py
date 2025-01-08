from collections.abc import Iterable
from typing import Unpack

from ...modules import AddMagnitude, NanToNum, NesterovMomentum, Normalize, Interpolate
from ...modules import RandomCoordinateMomentum as _RandomCoordinateMomentum
from ...modules import (Reciprocal, _CommonKwargs, _get_baked_in_and_module_lr,
                        _make_common_modules)
from ...modules.experimental.gradmin import GradMin as _GradMin
from ...modules.experimental.squared_grad_norm_fdm import \
    SquaredGradientNormFDM as _SquaredGradientNormFDM
from ..modular import Modular


class SquaredGradientNormFDM(Modular):
    """Experimental (maybe don't use yet)."""
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
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-2,
        **kwargs: Unpack[_CommonKwargs]
    ):
        lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs)
        main = [AddMagnitude(eps), Reciprocal(), NanToNum(0,0,0), Normalize(lr)]
        modules = _make_common_modules(main, lr_module = lr_module, kwargs=kwargs)
        super().__init__(params, modules)


class NestedNesterov(Modular):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentums: Iterable[float] = (0.5, 0.5, 0.5),
        dampening: float | Iterable[float] = 0,
        **kwargs: Unpack[_CommonKwargs] # type:ignore
    ):
        momentums = list(momentums)
        if isinstance(dampening, (int, float)): dampening = [dampening for _ in momentums]
        main = [NesterovMomentum(m, d) for m, d in zip(momentums, dampening)]
        modules = _make_common_modules(main, lr_module = lr, kwargs=kwargs)
        super().__init__(params, modules)

class RandomCoordinateMomentum(Modular):
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
