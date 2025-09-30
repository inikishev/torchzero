import itertools
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, Transform, step
from ...linalg.linear_operator import Dense
from ...utils import TensorList, vec_to_tensors_
from ...utils.derivatives import (
    flatten_jacobian,
    jacobian_wrt,
)
from ..second_order.newton import (
    _try_cholesky_solve,
    _least_squares_solve,
    _try_lu_solve,
)


class NewtonNewton(Transform):
    """Applies Newton-like preconditioning to Newton step.

    This is a method that I thought of and then it worked. Here is how it works:

    1. Calculate newton step by solving Hx=g

    2. Calculate jacobian of x wrt parameters and call it H2

    3. Solve H2 x2 = x for x2.

    4. Optionally, repeat (if order is higher than 3.)
    """
    def __init__(
        self,
        reg: float = 1e-6,
        order: int = 3,
        vectorize: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(order=order, reg=reg, vectorize=vectorize)
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        params = TensorList(objective.params)
        closure = objective.closure
        if closure is None: raise RuntimeError('NewtonNewton requires closure')

        reg = fs['reg']
        vectorize = fs['vectorize']
        order = fs['order']

        # ------------------------ calculate grad and hessian ------------------------ #
        P = None
        with torch.enable_grad():
            loss = objective.loss = objective.loss_approx = closure(False)
            g_list = torch.autograd.grad(loss, params, create_graph=True)
            objective.grads = list(g_list)

            xp = torch.cat([t.ravel() for t in g_list])
            I = torch.eye(xp.numel(), dtype=xp.dtype, device=xp.device)

            for o in range(2, order + 1):
                is_last = o == order
                H_list = jacobian_wrt([xp], params, create_graph=not is_last, batched=vectorize)
                with torch.no_grad() if is_last else nullcontext():
                    H = flatten_jacobian(H_list)
                    if reg != 0: H = H + I * reg
                    if P is None: P = H
                    else: P = P @ H

                    if not is_last:
                        x = _try_cholesky_solve(H, xp)
                        if x is None: x = _try_lu_solve(H, xp)
                        if x is None: x = _least_squares_solve(H, xp)
                        xp = x.squeeze()

        self.global_state["P"] = P

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()
        P = self.global_state['P']
        b = torch.cat([t.ravel() for t in updates])

        sol = _try_cholesky_solve(P, b)
        if sol is None: sol = _try_lu_solve(P, b)
        if sol is None: sol = _least_squares_solve(P, b)

        vec_to_tensors_(sol, updates)
        return objective

    @torch.no_grad
    def get_H(self, objective=...):
        return Dense(self.global_state["P"])
