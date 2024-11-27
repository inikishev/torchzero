import typing as T
from collections import abc

import torch

from ...modules import FDM as _FDM, SGD, ClosureOptimizerWrapper
from ...modules.gradient_approximation._fd_formulas import _FD_Formulas
from ..modular import ModularOptimizer


class FDM(ModularOptimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        momentum:float = 0,
        weight_decay:float = 0,
        dampening: float = 0,
        nesterov:bool = False,
        formula: _FD_Formulas = "forward",
        n_points: T.Literal[2, 3] = 2,
    ):
        """Gradient approximation via finite difference.

        Also known as Robbins-Monro stochastic approximation algorithm,
        or Kiefer–Wolfowitz algorithm with `central` formula.

        This performs `n + 1` evaluations per step with `forward` and `backward` formulas,
        and `2 * n` with `central` formula.

        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): learning rate. Defaults to 1e-3.
            eps (float, optional): finite difference epsilon. Defaults to 1e-3.
            momentum (float, optional): momentum factor. Defaults to 0.
            weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.
            dampening (float, optional): dampening for momentum. Defaults to 0.
            nesterov (bool, optional): enables Nesterov momentum (supports dampening). Defaults to False.
            formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
            n_points (T.Literal[2, 3], optional): number of points for finite difference formula, 2 or 3. Defaults to 2.
        """
        modules = [
            _FDM(eps = eps, formula=formula, n_points=n_points),
            SGD(lr = lr, momentum = momentum, weight_decay = weight_decay, dampening = dampening, nesterov = nesterov)
        ]
        super().__init__(params, modules)


class FDMWrapper(ModularOptimizer):
    def __init__(
        self,
        params,
        optimizer: torch.optim.Optimizer,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_points: T.Literal[2, 3] = 2,
    ):
        """Gradient approximation via finite difference. This wraps any other optimizer.
        This also supports optimizers that perform multiple gradient evaluations per step, like LBFGS.

        Exaple:
        ```
        lbfgs = torch.optim.LBFGS(params, lr = 1)
        fdm = FDMWrapper(params, optimizer = lbfgs)
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
        modules = [
            _FDM(eps = eps, formula=formula, n_points=n_points, make_closure=True),
            ClosureOptimizerWrapper(optimizer)
        ]
        super().__init__(params, modules)