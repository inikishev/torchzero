from abc import ABC, abstractmethod
import math
from collections import deque
from typing import Literal, Any

import torch
from ...core import Chainable, TensorwisePreconditioner

class _Solver:
    @abstractmethod
    def update(self, history: deque[torch.Tensor], damping: float | None) -> tuple[Any, Any]:
        """returns stuff for apply"""
    @abstractmethod
    def apply(self, __g: torch.Tensor, __A:torch.Tensor, __B:torch.Tensor) -> torch.Tensor:
        """apply preconditioning to tensor"""

class _SVDSolver(_Solver):
    def __init__(self, driver=None): self.driver=driver
    def update(self, history, damping):
        M_hist = torch.stack(tuple(history), dim=1)
        device = None # driver is CUDA only
        if self.driver is not None:
            device = M_hist.device
            M_hist = M_hist.cuda()

        try:
            U, S, _ = torch.linalg.svd(M_hist, full_matrices=False, driver=self.driver) # pylint:disable=not-callable

            if self.driver is not None:
                U = U.to(device); S = S.to(device)

            if damping is not None and damping != 0: S.add_(damping)
            return U, S

        except torch.linalg.LinAlgError:
            return None, None

    def apply(self, g: torch.Tensor, U: torch.Tensor, S: torch.Tensor):
        Utg = (U.T @ g).div_(S)
        return U @ Utg

class _SVDLowRankSolver(_Solver):
    def __init__(self, q: int = 6, niter: int = 2): self.q, self.niter = q, niter
    def update(self, history, damping):
        M_hist = torch.stack(tuple(history), dim=1)
        try:
            U, S, _ = torch.svd_lowrank(M_hist, q=self.q, niter=self.niter)
            if damping is not None and damping != 0: S.add_(damping)
            return U, S
        except torch.linalg.LinAlgError:
            return None, None

    def apply(self, g: torch.Tensor, U: torch.Tensor, S: torch.Tensor):
        Utg = (U.T @ g).div_(S)
        return U @ Utg

class _QRSolver(_Solver):
    def update(self, history, damping):
        M_hist = torch.stack(tuple(history), dim=1)
        try:
            Q, R = torch.linalg.qr(M_hist, mode='reduced') # pylint:disable=not-callable
            R_diag = R.diag().abs()
            if damping is not None and damping != 0: R_diag.add_(damping)
            return Q, R_diag
        except torch.linalg.LinAlgError:
            return None, None

    def apply(self, g: torch.Tensor, Q: torch.Tensor, R_diag: torch.Tensor):
        Qtg = (Q.T @ g).div_(R_diag)
        return Q @ Qtg


class _EighSolver(_Solver):
    def update(self, history, damping):
        M_hist = torch.stack(tuple(history), dim=1)
        grams = M_hist @ M_hist.T # (d, d)
        if damping is not None and damping != 0: grams.diagonal(dim1=-2, dim2=-1).add_(damping)
        try:
            L, Q = torch.linalg.eigh(grams) # L: (d,), Q: (d, d) # pylint:disable=not-callable
            return Q, L.abs().clamp_(min=1e-12)
        except torch.linalg.LinAlgError:
            return None, None

    def apply(self, g: torch.Tensor, Q: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        Qtg = (Q.T @ g).div_(L)
        return Q @ Qtg


SOLVERS = {
    "svd": _SVDSolver(), # fallbacks on "gesvd" which basically takes ages or just hangs completely
    "svd_gesvdj": _SVDSolver("gesvdj"), # no fallback on slow "gesvd"
    "svd_gesvda": _SVDSolver("gesvda"), # approximate method for wide matrices, sometimes better sometimes worse but faster
    "svd_lowrank": _SVDLowRankSolver(), # maybe need to tune parameters for this
    "eigh": _EighSolver(), # this is O(n**2) storage
    "qr": _QRSolver(),
}

def maybe_lerp_(state_: dict, beta: float | None, key, value: Any):
    if (key not in state_) or (beta is None) or (not isinstance(value, torch.Tensor)): state_[key] = value
    else:
        if state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class SpectralPreconditioner(TensorwisePreconditioner):
    """A low rank preconditioner via SVD on history of past gradients or gradient differences scaled by parameter differences.

    Args:
        history_size (int, optional): number of past gradients to store for preconditioning. Defaults to 10.
        update_freq (int, optional): how often to re-compute the preconditioner. Defaults to 1.
        damping (float, optional): damping term, makes it closer to GD. Defaults to 1e-7.
        order (int, optional):
            whitening order, 1 approximates FIM (maybe), 2 - hessian (maybe), 3+ - god knows what.
        solver (str, optional): what to use for whitening. Defaults to 'svd'.
        U_beta (float | None, optional): beta for U (probably a bad idea). Defaults to None.
        S_beta (float | None, optional): beta for S (probably a bad idea). Defaults to None.
        interval (int, optional): How often to update history. Defaults to 1 (every step).
        concat_params (bool, optional):
            whether to apply preconditioning to each tensor (False, default) or to all tensors concatenated into a vector (True). Latter will be slower but captures interactions between layers. Defaults to True.
        scale_first (bool, optional): makes first step small, usually not needed. Defaults to False.
        inner (Chainable | None, optional): Inner modules applied after updating preconditioner and before applying it. Defaults to None.
    """
    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-12,
        order: int = 1,
        solver: Literal['svd', 'svd_gesvdj', 'svd_gesvda', 'svd_lowrank', 'eigh', 'qr'] | _Solver = 'svd_gesvdj',
        U_beta: float | None = None,
        S_beta: float | None = None,
        interval: int = 1,
        concat_params: bool = False,
        scale_first: bool = False,
        inner: Chainable | None = None,
    ):
        if isinstance(solver, str): solver = SOLVERS[solver]
        # history is still updated each step so Precondition's update_freq has different meaning
        defaults = dict(history_size=history_size, update_freq=update_freq, damping=damping, order=order, U_beta=U_beta, S_beta=S_beta, solver=solver)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, scale_first=scale_first, inner=inner, update_freq=interval)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        order = settings['order']
        history_size = settings['history_size']
        update_freq = settings['update_freq']
        damping = settings['damping']
        U_beta = settings['U_beta']
        S_beta = settings['S_beta']
        solver: _Solver = settings['solver']

        if 'history' not in state: state['history'] = deque(maxlen=history_size)
        history = state['history']

        if order == 1: history.append(tensor.clone().view(-1))
        else:

            # if order=2, history is of gradient differences, order 3 is differences between differences, etc
            # normalized by parameter differences
            cur_p = param.clone()
            cur_g = tensor.clone()
            for i in range(1, order):
                if f'prev_g_{i}' not in state:
                    state[f'prev_p_{i}'] = cur_p
                    state[f'prev_g_{i}'] = cur_g
                    break
                else:
                    s_k = cur_p - state[f'prev_p_{i}']
                    y_k = cur_g - state[f'prev_g_{i}']
                    state[f'prev_p_{i}'] = cur_p
                    state[f'prev_g_{i}'] = cur_g
                    cur_p = s_k
                    cur_g = y_k

                if i == order - 1:
                    cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=1e-8) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))

        step = state.get('step', 0)
        if step % update_freq == 0 and len(history) != 0:
            A, B = solver.update(history, damping=damping)
            maybe_lerp_(state, U_beta, 'A', A)
            maybe_lerp_(state, S_beta, 'B', B)

        if len(history) != 0:
            state['step'] = step + 1 # do not increment if no history (gathering s_ks and y_ks)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        history_size = settings['history_size']
        solver: _Solver = settings['solver']

        A = state.get('A', None)
        if A is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.div_(max(1, tensor.abs().sum())) # pyright:ignore[reportArgumentType]

        B = state['B']
        update = solver.apply(tensor.view(-1), A, B).view_as(tensor)

        n = len(state['history'])
        if n != history_size: update.mul_(n/history_size)
        return update

