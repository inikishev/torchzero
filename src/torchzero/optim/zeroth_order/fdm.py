from typing import Literal

import torch

from ...modules import FDM as _FDM, WrapClosure, SGD, WeightDecay, LR
from ...modules.gradient_approximation._fd_formulas import _FD_Formulas
from ..modular import Modular


class FDM(Modular):
    """Gradient approximation via finite difference.

    This performs `n + 1` evaluations per step with `forward` and `backward` formulas,
    and `2 * n` with `central` formula, where n is the number of parameters.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_points (T.Literal[2, 3], optional): number of points for finite difference formula, 2 or 3. Defaults to 2.
        momentum (float, optional): momentum. Defaults to 0.
        dampening (float, optional): momentum dampening. Defaults to 0.
        nesterov (bool, optional):
            enables nesterov momentum, otherwise uses heavyball momentum. Defaults to False.
        weight_decay (float, optional): weight decay (L2 regularization). Defaults to 0.
        decoupled (bool, optional):
            decouples weight decay from gradient. If True, weight decay doesn't depend on learning rate.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_points: Literal[2, 3] = 2,
        momentum: float = 0,
        dampening: float = 0,
        nesterov: bool = False,
        weight_decay: float = 0,
        decoupled=False,

    ):
        modules: list = [
            _FDM(eps = eps, formula=formula, n_points=n_points),
            SGD(momentum = momentum, dampening = dampening, weight_decay = weight_decay if not decoupled else 0, nesterov = nesterov),
            LR(lr),

        ]
        if decoupled: modules.append(WeightDecay(weight_decay))
        super().__init__(params, modules)


class FDMWrapper(Modular):
    """Gradient approximation via finite difference. This wraps any other optimizer.
    This also supports optimizers that perform multiple gradient evaluations per step, like LBFGS.

    Exaple:
    ```
    lbfgs = torch.optim.LBFGS(params, lr = 1)
    fdm = FDMWrapper(optimizer = lbfgs)
    ```

    This performs n+1 evaluations per step with `forward` and `backward` formulas,
    and 2*n with `central` formula.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        optimizer (torch.optim.Optimizer): optimizer that will perform optimization using FDM-approximated gradients.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_points (T.Literal[2, 3], optional): number of points for finite difference formula, 2 or 3. Defaults to 2.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_points: Literal[2, 3] = 2,
    ):
        modules = [
            _FDM(eps = eps, formula=formula, n_points=n_points, target = 'closure'),
            WrapClosure(optimizer)
        ]
        # some optimizers have `eps` setting in param groups too.
        # it should not be passed to FDM
        super().__init__([p for g in optimizer.param_groups.copy() for p in g['params']], modules)