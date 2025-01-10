import torch

from ... import tl
from ...core import OptimizationState
from .ls_base import LineSearchBase


class ArmijoLS(LineSearchBase):
    """Armijo backtracking line search

    Args:
        lr (float): initial, maximal lr.
        mul (float, optional): lr multiplier on each iteration. Defaults to 0.5.
        beta (float, optional):
            armijo condition parameter, fraction of expected linear loss decrease to accept.
            Larger values mean loss needs to decrease more for a step sizer to be accepted. Defaults to 1e-4.
        max_iter (int, optional): maximum iterations. Defaults to 10.
        log_lrs (bool, optional): logs learning rates. Defaults to False.
    """
    def __init__(
        self,
        lr: float,
        mul: float = 0.5,
        beta: float = 1e-2,
        max_iter: int = 10,
        log_lrs = False,
    ):
        defaults = dict(lr=lr)
        super().__init__(defaults, make_closure=False, maxiter=None, log_lrs=log_lrs)
        self.mul = mul
        self.beta = beta
        self.max_iter = max_iter

    @torch.no_grad
    def _find_best_lr(self, state: OptimizationState, params: tl.TensorList) -> float:
        if state.closure is None: raise ValueError("closure is not set")
        ascent = state.maybe_use_grad_(params)
        grad = state.maybe_compute_grad_(params)
        lr = self.get_first_group_key('lr')
        if state.fx0 is None: state.fx0 = state.closure(False)

        # loss decrease per lr=1 if function was linear
        decrease_per_lr = (grad*ascent).total_sum()

        for _ in range(self.max_iter):
            loss = self._evaluate_lr_(lr, state.closure, ascent, params)

            # expected decrease
            expected_decrease = decrease_per_lr * lr

            if (state.fx0 - loss) / expected_decrease >= self.beta:
                return lr

            lr *= self.mul

        return 0
