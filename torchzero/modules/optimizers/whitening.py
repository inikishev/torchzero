import math
from collections import deque
from typing import Literal

import torch

from ...core import Chainable, Precondition, TensorwisePreconditioner


def get_US(history: deque, damping: float):
    M_hist = torch.stack(tuple(history), dim=1)
    try:
        # U - (d, history_size)
        # S - (history_size, history_size)
        U, S, _ = torch.linalg.svd(M_hist, full_matrices=False) # pylint:disable=not-callable
        return U, S.pow_(2).add_(damping).sqrt_() # this is a more "correct" way to do damping

    except torch.linalg.LinAlgError:
        return None, None


def whiten(tensor: torch.Tensor, U: torch.Tensor, S: torch.Tensor, ):
    Utg = (U.T @ tensor).div_(S)
    return U @ Utg

def maybe_lerp_(state_: dict, beta: float | None, key, value: torch.Tensor):
    if key not in state_: state_[key] = value
    else:
        if beta is None or state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class HistoryPreconditioner(TensorwisePreconditioner):
    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-10,
        U_beta: float | None = None,
        S_beta: float | None = None,
        order: int = 1,
    ):
        super().__init__()
        self.history_size = history_size
        self.update_freq = update_freq # history is still updated each step so Precondition's update_freq has different meaning
        self.damping = damping
        self.U_beta = U_beta
        self.S_beta = S_beta
        self.order = order

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state):
        if 'history' not in state: state['history'] = deque(maxlen=self.history_size)
        history = state['history']

        if self.order == 1: history.append(tensor.clone().view(-1))
        else:

            # if order=2, history is of gradient differences, order 3 is differences between differences, etc
            # normalized by parameter differences
            cur_p = param.clone()
            cur_g = tensor.clone()
            for i in range(1, self.order):
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

                if i == self.order - 1:
                    cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=1e-8) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))


        step = state.get('step', 0)
        if step % self.update_freq == 0 and len(history) != 0:
            U, S = get_US(history, damping=self.damping)
            if U is not None and S is not None:
                maybe_lerp_(state, self.U_beta, 'U', U)
                maybe_lerp_(state, self.S_beta, 'S', S)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state):
        state['step'] = state.get('step', 0)
        n = len(state['history'])
        if n != 0: state['step'] += 1 # do not increment if history wasn't updated

        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.div_(max(1, tensor.abs().sum())) # pyright:ignore[reportArgumentType]

        S = state['S']
        update = whiten(tensor.view(-1), U, S).view_as(tensor)

        if n != self.history_size: update.mul_(n/self.history_size)
        return update



class WhitenViaSVD(Precondition):
    """Precondition using history of past gradients. Apparently chaining multiple of those might improve preconditioning accuracy.

    Args:
        history_size (int, optional): number of past gradients to store for preconditioning. Defaults to 10.
        update_freq (int, optional): how often to re-compute the preconditioner. Defaults to 1.
        damping (float, optional): damping term, makes it closer to GD. Defaults to 1e-7.
        eps (float, optional): minimal value for S. Defaults to 1e-8.
        order (int, optional):
            whitening order, 1 approximates FIM (maybe), 2 - hessian (maybe), 3+ - god knows what.
        U_beta (float | None, optional): beta for U (probably a bad idea). Defaults to None.
        Sv_beta (float | None, optional): beta for Sv (probably a bad idea). Defaults to None.
        update_into_precond (float | None, optional):
            whether to use history of previopus updates for whitening (probably a bad idea). Defaults to None.
        tensorwise (bool, optional):
            whether to apply preconditioning to each tensor or to all tensors concatenated into a vector. Latter will be slower but captures interactions between layers. Defaults to True.
        scale_first (bool, optional): makes first step small, usually not needed. Defaults to False.
        inner (Chainable | None, optional): Inner modules applied after updating preconditioner and before applying it. Defaults to None.
    """
    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-10,
        order: int = 1,
        U_beta: float | None = None,
        S_beta: float | None = None,
        tensorwise: bool = True,
        scale_first: bool = False,
        inner: Chainable | None = None,
    ):
        super().__init__(
            HistoryPreconditioner(history_size=history_size, update_freq=update_freq, damping=damping, U_beta=U_beta, S_beta=S_beta, order=order),
            uses_grad=False,
            tensorwise=tensorwise,
            scale_first=scale_first,
            inner = inner
        )