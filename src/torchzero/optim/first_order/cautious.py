from typing import Literal, Unpack

from ...core import OptimizerModule
from ...modules import Cautious, Adam, SGD, Lion, WeightDecay, LR
from ..modular import Modular


class CautiousAdamW(Modular):
    """Adam, but updates for parameters where update and gradient sign is inconsistent are negated.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): exponential decay rate of gradient moving average. Defaults to 0.9.
        beta2 (float, optional): exponential decay rate of squared gradient moving average. Defaults to 0.999.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        amsgrad (bool, optional):
            whether to use the AMSGrad variant of this algorithm from
            On the Convergence of Adam and Beyond (default: False).
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        c_eps (float, optional): epsilon for normalization after applying cautioning mask. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "negate" - negate them (same as using update magnitude and gradient sign).
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
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
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list[OptimizerModule] = [
            Adam(beta1 = beta1, beta2 = beta2, eps = eps, amsgrad = amsgrad),
            LR(lr),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))
        super().__init__(params, modules)


class CautiousSGD(Modular):
    """SGD with momentum, but updates for parameters where update and gradient sign is inconsistent are negated.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        c_eps (float, optional): epsilon for normalization after applying cautioning mask. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "negate" - negate them (same as using update magnitude and gradient sign).
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        nesterov: bool = True,
        c_eps = 1e-6,
        normalize = False,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list[OptimizerModule] = [
            SGD(momentum = momentum, dampening = dampening, weight_decay = 0, nesterov = nesterov),
            LR(lr),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]

        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))

        super().__init__(params, modules)


class CautiousLion(Modular):
    """Lion optimizer, but updates for parameters where update and gradient sign is inconsistent are negated.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float, optional): dampening for momentum. Defaults to 0.9.
        beta2 (float, optional): momentum factor. Defaults to 0.99.
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        c_eps (float, optional): epsilon for normalization after applying cautioning mask. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "negate" - negate them (same as using update magnitude and gradient sign).
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        c_eps = 1e-6,
        normalize = False,
        mode: Literal['zero', 'grad', 'backtrack'] = 'zero',
        weight_decay: float = 0,
        decoupled=True,
    ):
        modules: list[OptimizerModule] = [
            Lion(beta1, beta2),
            LR(lr),
            Cautious(normalize = normalize, eps = c_eps, mode = mode),
        ]

        if decoupled: modules.append(WeightDecay(weight_decay))
        else: modules.insert(0, WeightDecay(weight_decay))

        super().__init__(params, modules)
