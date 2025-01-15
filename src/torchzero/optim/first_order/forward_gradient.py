from typing import Literal

import torch

from ...core import OptimizationState, OptimizerModule
from ...modules import ForwardGradient as _ForwardGradient, SGD, WeightDecay, LR
from ...tensorlist import Distributions
from ..modular import Modular


class ForwardGradient(Modular):
    """

    Evaluates jacobian-vector product with a random vector using forward mode autodiff (torch.func.jvp), which is
    the true directional derivative in the direction of that vector.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        n_samples (int): number of forward gradients to evaluate and average.
        distribution (Distributions): distribution for random tangent vector.
        mode (str):
            "jvp" - uses forward mode AD, usually slightly slower than backward mode AD  but uses significantly less memory.

            "grad" - evaluates gradient with `loss.backward()` which may be faster but uses all the memory, mainly useful for
            benchmarking as there is probably no point in forward gradient if full gradient is available.

            "fd" - uses finite difference to estimate JVP, doesn't require gradients to be known. Equivalent to randomized FDM.

        fd_eps (float, optional): epsilon for finite difference, only has effect if mode is "fd". Defaults to 1e-4.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.

    Reference:
        Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022).
        Gradients without backpropagation. arXiv preprint arXiv:2202.08587.
        https://arxiv.org/abs/2202.08587
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        n_samples: int = 1,
        distribution: Distributions = "normal",
        mode: Literal["jvp", "grad", "fd"] = "jvp",
        fd_eps: float = 1e-4,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,
    ):
        modules: list = [
            _ForwardGradient(
                n_samples=n_samples,
                distribution=distribution,
                mode=mode,
                fd_eps=fd_eps,
            ),
            SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            LR(lr),
        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)