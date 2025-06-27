from collections import deque
from typing import Literal, Any
import warnings

import torch
from ...core import Chainable, TensorwiseTransform

def lm_adagrad_update_preconditioner(history: deque[torch.Tensor], damping, rdamping, true_damping: bool, centered:bool):
    M = torch.stack(tuple(history), dim=1)

    if centered:
        # centering is unstable when not enough history has been recorded
        maxlen = history.maxlen if history.maxlen is not None else 5
        if len(history) >= min(maxlen, 5):
            M -= M.mean(1, keepdim=True)

    device = M.device
    if torch.cuda.is_available(): M = M.cuda()

    try:
        if torch.cuda.is_available():
            U, S, _ = torch.linalg.svd(M, full_matrices=False, driver='gesvda') # pylint:disable=not-callable
        else:
            warnings.warn("CUDA is not available, cuSOLVER's \"gesvda\" can't be used so LMAdagrad may be significantly slower.")
            U, S, _ = torch.linalg.svd(M, full_matrices=False) # pylint:disable=not-callable

        U = U.to(device); S = S.to(device)

        if damping != 0 or rdamping != 0:
            if rdamping != 0: rdamping *= torch.linalg.vector_norm(S) # pylint:disable=not-callable
            Iu = damping + rdamping
            if true_damping:
                S.pow_(2)
                Iu **= 2
            S.add_(Iu)
            if true_damping: S.sqrt_()

        return U, S

    except torch.linalg.LinAlgError:
        return None, None

def lm_adagrad_apply(g: torch.Tensor, U: torch.Tensor, S: torch.Tensor):
    Z = (U.T @ g)/S
    return U @ Z


def maybe_lerp_(state_: dict, beta: float | None, key, value: Any):
    if (key not in state_) or (beta is None) or (not isinstance(value, torch.Tensor)): state_[key] = value
    else:
        if state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class LMAdagrad(TensorwiseTransform):
    """
    Limited-memory full matrix Adagrad.

    The update rule is to stack recent gradients into M, compute U, S <- SVD(M), then calculate update as U ((Uᵀg)/S).

    This is equivalent to full-matrix Adagrad on recent gradients.

    Args:
        history_size (int, optional): number of past gradients to store. Defaults to 10.
        update_freq (int, optional): frequency of updating the preconditioner (U and S). Defaults to 1.
        damping (float, optional): damping value. Defaults to 1e-4.
        rdamping (float, optional): value of damping relative to singular values norm. Defaults to 0.
        order (int, optional):
            order=2 means gradient differences are used in place of gradients. Higher order uses higher order differences. Defaults to 1.
        true_damping (bool, optional):
            If True, damping is added to squared singular values to mimic Adagrad. Defaults to True.
        centered (bool, optional):
            if True, centers observations by mean of each feature before calculating the covariance matrix, which is how you are supposed to calculate it but it is more unstable. Defaults to False.
        U_beta (float | None, optional): momentum for U (too unstable, don't use). Defaults to None.
        S_beta (float | None, optional): momentum for S (too unstable, don't use). Defaults to None.
        interval (int, optional): Interval between gradients that are added to history (2 means every second gradient is used). Defaults to 1.
        concat_params (bool, optional): if True, treats all parameters as a single vector, meaning it will also whiten inter-parameters. Defaults to True.
        inner (Chainable | None, optional): preconditioner will be applied to output of this module. Defaults to None.

    Examples:
        Limited-memory Adagrad

        .. code-block:: python

            optimizer = tz.Modular(
                model.parameters(),
                tz.m.LMAdagrad(),
                tz.m.LR(0.1)
            )

        Adam with L-Adagrad preconditioner (for debiasing second beta is 0.999 arbitrarily)

        .. code-block:: python

            optimizer = tz.Modular(
                model.parameters(),
                tz.m.LMAdagrad(inner=tz.m.EMA()),
                tz.m.Debias(0.9, 0.999),
                tz.m.LR(0.01)
            )

        Stable Adam with L-Adagrad preconditioner (this is what I would recommend)

        .. code-block:: python

            optimizer = tz.Modular(
                model.parameters(),
                tz.m.LMAdagrad(inner=tz.m.EMA()),
                tz.m.Debias(0.9, 0.999),
                tz.m.ClipNormByEMA(max_ema_growth=1.2),
                tz.m.LR(0.01)
            )
    """

    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-4,
        rdamping: float = 0,
        order: int = 1,
        true_damping: bool = True,
        centered: bool = False,
        U_beta: float | None = None,
        S_beta: float | None = None,
        interval: int = 1,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        # history is still updated each step so Precondition's update_freq has different meaning
        defaults = dict(history_size=history_size, update_freq=update_freq, damping=damping, rdamping=rdamping, centered=centered, true_damping=true_damping, order=order, U_beta=U_beta, S_beta=S_beta)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, inner=inner, update_freq=interval)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, loss, state, settings):
        order = settings['order']
        history_size = settings['history_size']
        update_freq = settings['update_freq']
        damping = settings['damping']
        rdamping = settings['rdamping']
        true_damping = settings['true_damping']
        U_beta = settings['U_beta']
        S_beta = settings['S_beta']
        centered = settings['centered']

        if 'history' not in state: state['history'] = deque(maxlen=history_size)
        history = state['history']

        if order == 1:
            t = tensor.clone().view(-1)
            history.append(t)
        else:

            # if order=2, history is of gradient differences, order 3 is differences between differences, etc
            # scaled by parameter differences
            cur_p = param.clone()
            cur_g = tensor.clone()
            for i in range(1, order):
                if f'prev_g_{i}' not in state:
                    state[f'prev_p_{i}'] = cur_p
                    state[f'prev_g_{i}'] = cur_g
                    break

                s = cur_p - state[f'prev_p_{i}']
                y = cur_g - state[f'prev_g_{i}']
                state[f'prev_p_{i}'] = cur_p
                state[f'prev_g_{i}'] = cur_g
                cur_p = s
                cur_g = y

                if i == order - 1:
                    cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=1e-8) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))

        step = state.get('step', 0)
        if step % update_freq == 0 and len(history) != 0:
            U, S = lm_adagrad_update_preconditioner(history, damping=damping, rdamping=rdamping,
                                                       true_damping=true_damping, centered=centered)
            maybe_lerp_(state, U_beta, 'U', U)
            maybe_lerp_(state, S_beta, 'S', S)

        if len(history) != 0:
            state['step'] = step + 1 # do not increment if no history (gathering s_ks and y_ks)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):

        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.clip_(-0.1, 0.1) # pyright:ignore[reportArgumentType]

        S = state['S']
        update = lm_adagrad_apply(tensor.view(-1), U, S).view_as(tensor)

        return update

