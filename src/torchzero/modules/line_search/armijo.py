import torch

from ...tensorlist import TensorList
from ...core import OptimizationVars
from .base_ls import LineSearchBase


class ArmijoLS(LineSearchBase):
    """Armijo backtracking line search

    Args:
        alpha (float): initial step size.
        mul (float, optional): lr multiplier on each iteration. Defaults to 0.5.
        beta (float, optional):
            armijo condition parameter, fraction of expected linear loss decrease to accept.
            Larger values mean loss needs to decrease more for a step sizer to be accepted. Defaults to 1e-4.
        max_iter (int, optional): maximum iterations. Defaults to 10.
        log_lrs (bool, optional): logs learning rates. Defaults to False.
    """
    def __init__(
        self,
        alpha: float = 1,
        mul: float = 0.5,
        beta: float = 1e-2,
        max_iter: int = 10,
        log_lrs = False,
    ):
        defaults = dict(alpha=alpha)
        super().__init__(defaults, maxiter=None, log_lrs=log_lrs)
        self.mul = mul
        self.beta = beta
        self.max_iter = max_iter

    @torch.no_grad
    def _find_best_lr(self, vars: OptimizationVars, params: TensorList) -> float:
        if vars.closure is None: raise RuntimeError(f"Line searches ({self.__class__.__name__}) require a closure")
        ascent = vars.maybe_use_grad_(params)
        grad = vars.maybe_compute_grad_(params)
        alpha = self.get_first_group_key('alpha')
        if vars.fx0 is None: vars.fx0 = vars.closure(False)

        # loss decrease per lr=1 if function was linear
        decrease_per_lr = (grad*ascent).total_sum()

        for _ in range(self.max_iter):
            loss = self._evaluate_lr_(alpha, vars.closure, ascent, params)

            # expected decrease
            expected_decrease = decrease_per_lr * alpha

            if (vars.fx0 - loss) / expected_decrease >= self.beta:
                return alpha

            alpha *= self.mul

        return 0
