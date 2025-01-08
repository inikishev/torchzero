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
    """for experiments, unlikely to work well on most problems.

    explanation - this should equate to newton method with just 2 backward passes, but only if hessian is purely diagonal"""
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

    explanation - this basically uses normalized (1 / gradient), adds epsilon to gradient magnitude."""
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
