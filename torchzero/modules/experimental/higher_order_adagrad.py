import itertools
import math
import warnings
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import scipy.optimize
import torch

from ...core import Chainable, Modular, Module, apply_transform
from ...utils import TensorList, vec_to_tensors, vec_to_tensors_
from ..line_search import Backtracking
from ..quasi_newton import LBFGS

_LETTERS = 'abcdefghijklmnopqrstuvwxy'

class HigherOrderAdagrad(Module):
    """
    .. note::
        Conceptual.

    .. note::
        Extremely expensive.

    .. note::
        Doesn't work.
    """
    def __init__(
        self,
        order: int = 3,
        history_size: int | None = 100,
        iters: int = 100,
        solver = lambda p: Modular(p, LBFGS(), Backtracking())
    ):
        defaults = dict(order=order, history_size=history_size, solver=solver, iters=iters)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        g = TensorList(var.get_grad()).to_vec()
        settings = self.settings[params[0]]

        order = settings['order']
        history_size = settings['history_size']
        solver = settings['solver']
        iters = settings['iters']

        history = self.global_state.setdefault('history', deque(maxlen=history_size))
        history.append(g)

        whitener = torch.eye(g.numel(), device=g.device, dtype=g.dtype)

        opt = solver([whitener.requires_grad_()])
        G = torch.stack(tuple(history), 0) # n, ndim

        def objective(backward=True):
            G_white = G @ whitener # n, ndim

            loss = cast(torch.Tensor, 0)
            with torch.enable_grad() if backward else nullcontext():
                for o in range(2, order+1):
                    l = _LETTERS[:o]
                    s1 = ',z'.join(l)
                    # batched outer product
                    # za,zb,zc->abc
                    cov = torch.einsum(f'z{s1}->{l}', *(G_white for _ in range(o)))

                    diag_indices = torch.arange(g.numel())

                    diag = cov[*(diag_indices for _ in range(o))]
                    loss = loss + (diag - 1).pow(2).mean()
                    loss = loss + (cov.pow(2).sum() - diag.pow(2).sum()) / (g.numel() ** 2)


                if backward:
                    whitener.grad = torch.autograd.grad(loss, whitener)[0]

            return loss

        prev_loss = None
        for i in range(iters):
            loss = opt.step(objective)

            if prev_loss is not None and abs(prev_loss - loss) < 1e-6:
                break

            prev_loss = loss

        g_precond = g @ whitener
        var.update = vec_to_tensors(g_precond, params)
        return var

