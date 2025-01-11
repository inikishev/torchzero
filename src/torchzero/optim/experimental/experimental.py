from collections.abc import Iterable
from typing import Literal, Unpack

from ...modules import (
    LR,
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
    Multistep,
    NanToNum,
    NesterovMomentum,
    Normalize,
    RDiv,
    Reciprocal,
    ReduceOutwardLR,
    WeightDecay,
)
from ...modules import RandomCoordinateMomentum as _RandomCoordinateMomentum
from ...modules.experimental import GradMin as _GradMin
from ...modules.experimental import (
    HVPDiagNewton as _HVPDiagNewton,
)
from ...modules.experimental import MinibatchRprop as _MinibatchRprop
from ..modular import Modular


class HVPDiagNewton(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this should approximate newton method with 2 backward passes, but only if hessian is purely diagonal"""
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        eps: float = 1e-2,
    ):
        modules = [_HVPDiagNewton(eps = eps), LR(lr)]
        super().__init__(params, modules)


class ReciprocalSGD(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this basically uses normalized *1 / (gradient + eps)*."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-2,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            AddMagnitude(eps, add_to_zero=False),
            Reciprocal(),
            NanToNum(0,0,0),
            Normalize(1),
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))

        super().__init__(params, modules)


class MomentumNumerator(Modular):
    """for experiments, unlikely to work well on most problems. (somewhat promising)

    explanation - momentum divided by gradient."""
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        eps: float = 1e-2,
        weight_decay: float = 0,
        decoupled=True,    ):

        modules: list = [
            Divide(
                numerator = SGD(lr = 1, momentum = momentum, nesterov=nesterov),
                denominator=[Abs(), Add(eps)]
            ),
            Normalize(lr)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            Div([SGD(lr = 1, momentum=momentum, nesterov=nesterov), Abs(), Add(eps), Normalize(1)]),
            Normalize(lr)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        weight_decay: float = 0,
        decoupled=True,
    ):

        modules: list = [
            Interpolate(HeavyBall(momentum, dampening), NesterovMomentum(momentum, dampening), strength),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class ExtraCautiousAdam(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - caution with true backtracking."""
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
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            Adam(lr, beta1, beta2, eps, amsgrad),
            Lerp(Cautious(normalize, c_eps, mode), strength),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        nesterov: bool = False,
        mul = 0.5,
        use_grad=False,
        invert=False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
            ReduceOutwardLR(mul, use_grad, invert)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class MultistepSGD(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - perform multiple steps per batch. Momentum applies to the total update over multiple step"""
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        num_steps=2,
        weight_decay: float = 0,
        decoupled=True,
    ):
        # lr, lr_module = _get_baked_in_and_module_lr(lr, kwargs) # multistep must use lr

        modules: list = [
            Multistep(LR(lr), num_steps=num_steps),
            SGD(lr = 1, momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)


class MinibatchRprop(Modular):
    """
    for experiments, unlikely to work well on most problems.

    explanation: does 2 steps per batch, applies rprop rule on the second step.
    """
    def __init__(
        self,
        params,
        lr: float = 1,
        nplus: float = 1.2,
        nminus: float = 0.5,
        lb: float | None = 1e-6,
        ub: float | None = 50,
        backtrack=True,
        next_mode = 'continue',
        increase_mul = 0.5,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _MinibatchRprop(lr, nplus=nplus,nminus=nminus,lb=lb,ub=ub,backtrack=backtrack,next_mode=next_mode,increase_mul=increase_mul)
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
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
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [_RandomCoordinateMomentum(p, nesterov), LR(lr)]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)

class GradMin(Modular):
    """for experiments, unlikely to work well on most problems.

    explanation - this uses gradient wrt sum of gradients + loss."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        loss_term: float = 1,
        square: bool = False,
        maximize_grad: bool = False,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list = [
            _GradMin(loss_term, square, maximize_grad),
            SGD(lr = lr, momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),

        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)


