from collections import deque

import torch

from ...core import TensorwisePreconditioner, Precondition, Chainable


def get_U_Sv(history: deque, damping: float, eps: float):
    M_hist = torch.stack(tuple(history), dim=1)
    try:
        # U - (d, history_size)
        # S - (history_size, history_size)
        U, S, _ = torch.linalg.svd(M_hist, full_matrices=False) # pylint:disable=not-callable

        Sv = (S**2 / len(history)) + damping
        Sv = torch.max(Sv, torch.full_like(Sv, eps))

        Sv = 1.0 / torch.sqrt(Sv)
        return U, Sv

    except torch.linalg.LinAlgError:
        return None, None


def apply_svd_preconditioner(tensor: torch.Tensor, U: torch.Tensor, Sv: torch.Tensor, ):
    Utg = (U.T @ tensor) * Sv
    return U @ Utg

def maybe_lerp_(state_: dict, beta: float | None, key, value: torch.Tensor):
    if key not in state_: state_[key] = value
    else:
        if beta is None or state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class SVDHistoryPreconditioner(TensorwisePreconditioner):
    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-5,
        eps: float = 1e-7,
        U_beta: float | None = None,
        Sv_beta: float | None = None,
        update_into_precond: bool = False,
    ):
        super().__init__()
        self.history_size = history_size
        self.update_freq = update_freq # history is still updated each step so Precondition's update_freq has different meaning
        self.damping = damping
        self.eps = eps
        self.U_beta = U_beta
        self.Sv_beta = Sv_beta
        self.update_into_precond = update_into_precond

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state):
        if 'history' not in state: state['history'] = deque(maxlen=self.history_size)
        if (not self.update_into_precond) or len(state['history']) == 0: state['history'].append(tensor.clone().view(-1))
        step = state.get('step', 0)
        if step % self.update_freq == 0:
            U, Sv = get_U_Sv(state['history'], self.damping, self.eps)
            if U is not None and Sv is not None:
                maybe_lerp_(state, self.U_beta, 'U', U)
                maybe_lerp_(state, self.Sv_beta, 'Sv', Sv)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state):
        state['step'] = state.get('step', 0) + 1

        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.div_(max(1, tensor.abs().sum())) # pyright:ignore[reportArgumentType]

        Sv = state['Sv']
        update = apply_svd_preconditioner(tensor.view(-1), U, Sv).view_as(tensor)

        if self.update_into_precond:
            if state['step'] == 1: del state['history'][0] # clear grad one
            state['history'].append(update.view(-1).clone())

        return update


class SVDHistoryWhiten(Precondition):
    """Precondition using history of past gradients. You can chain multiple of those to improve preconditioning accuracy.

    Args:
        history_size (int, optional): number of past gradients to store for preconditioning. Defaults to 10.
        update_freq (int, optional): how often to re-compute the preconditioner. Defaults to 1.
        damping (float, optional): makes it closer to GD. Defaults to 1e-5.
        eps (float, optional): epsilon for division. Defaults to 1e-7.
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
        damping: float = 1e-5,
        eps: float = 1e-7,
        U_beta: float | None = None,
        Sv_beta: float | None = None,
        update_into_precond: bool = False,
        tensorwise: bool = True,
        scale_first: bool = False,
        inner: Chainable | None = None,
    ):
        super().__init__(
            SVDHistoryPreconditioner(history_size=history_size, update_freq=update_freq, damping=damping, eps=eps, U_beta=U_beta, Sv_beta=Sv_beta, update_into_precond=update_into_precond),
            uses_grad=False,
            tensorwise=tensorwise,
            scale_first=scale_first,
            inner = inner
        )