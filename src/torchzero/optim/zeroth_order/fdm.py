from typing import Literal, Unpack

import torch

from ...modules import FDM as _FDM, OptimizerWrapper, _make_common_modules, _CommonKwargs
from ...modules.gradient_approximation._fd_formulas import _FD_Formulas
from ..modular import Modular


class FDM(Modular):
    """Gradient approximation via finite difference.

    Also known as Robbins-Monro stochastic approximation algorithm,
    or Kiefer–Wolfowitz algorithm with `central` formula.

    This performs `n + 1` evaluations per step with `forward` and `backward` formulas,
    and `2 * n` with `central` formula.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. Defaults to 1e-3.
        eps (float, optional): finite difference epsilon. Defaults to 1e-3.
        formula (_FD_Formulas, optional): finite difference formula. Defaults to "forward".
        n_points (T.Literal[2, 3], optional): number of points for finite difference formula, 2 or 3. Defaults to 2.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        eps: float = 1e-3,
        formula: _FD_Formulas = "forward",
        n_points: Literal[2, 3] = 2,
        **kwargs: Unpack[_CommonKwargs],
    ):
        main = _FDM(eps = eps, formula=formula, n_points=n_points)
        modules = _make_common_modules(main, lr, kwargs)
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
            _FDM(eps = eps, formula=formula, n_points=n_points, make_closure=True),
            OptimizerWrapper(optimizer, pass_closure=True)
        ]
        super().__init__(optimizer.param_groups, modules)