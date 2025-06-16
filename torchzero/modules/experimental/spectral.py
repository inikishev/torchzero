from abc import ABC, abstractmethod
import math
from collections import deque
from typing import Literal, Any
import itertools

import torch
from ...core import Chainable, TensorwiseTransform
from ...utils.linalg.matrix_funcs import matrix_power_eigh
from ...utils.linalg.svd import randomized_svd
from ...utils.linalg.qr import qr_householder

def spectral_update(history, damping, rdamping, true_damping: bool):
    M_hist = torch.stack(tuple(history), dim=1)
    device = M_hist.device
    M_hist = M_hist.cuda()

    try:
        U, S, _ = torch.linalg.svd(M_hist, full_matrices=False, driver='gesvda') # pylint:disable=not-callable
        U = U.to(device); S = S.to(device)

        if damping != 0 or rdamping != 0:
            if rdamping != 0: rdamping *= torch.linalg.vector_norm(S) # pylint:disable=not-callable
            Iu = damping + rdamping
            if true_damping:
                S.pow_(2)
                Iu **= 2
            S.add_(Iu)
            if true_damping: S.sqrt_()

        return U, 1/S

    except torch.linalg.LinAlgError:
        return None, None

def spectral_apply(g: torch.Tensor, U: torch.Tensor, S_inv: torch.Tensor):
    Utg = (U.T @ g)*S_inv
    return U @ Utg


def maybe_lerp_(state_: dict, beta: float | None, key, value: Any):
    if (key not in state_) or (beta is None) or (not isinstance(value, torch.Tensor)): state_[key] = value
    else:
        if state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class SpectralPreconditioner(TensorwiseTransform):
    """
    The update rule is to stack recent gradients into M, compute U, S <- SVD(M), then calculate U (Uᵀg)/S.
    This is equivalent to full matrix Adagrad with accumulator initialized to zeros,
    except only recent :code:`history_size` gradients are used.
    However this doesn't require N^2 memory and is computationally less expensive than Shampoo.

    Args:
        history_size (int, optional): number of past gradients to store. Defaults to 10.
        update_freq (int, optional): frequency of updating the preconditioner (U and S). Defaults to 1.
        damping (float, optional): damping value. Defaults to 1e-4.
        rdamping (float, optional): value of damping relative to singular values norm. Defaults to 0.
        order (int, optional):
            order=2 means gradient differences are used in place of gradients. Higher order uses higher order differences. Defaults to 1.
        true_damping (bool, optional):
            If True, damping is added to squared singular values to mimic Adagrad. Defaults to True.
        U_beta (float | None, optional): momentum for U (too unstable, don't use). Defaults to None.
        S_beta (float | None, optional): momentum for 1/S (too unstable, don't use). Defaults to None.
        interval (int, optional): Interval between gradients that are added to history (2 means every second gradient is used). Defaults to 1.
        concat_params (bool, optional): if True, treats all parameters as a single vector, meaning it will also whiten inter-parameters. Defaults to False.
        normalize (bool, optional): whether to normalize gradients, this doesn't work well so don't use it. Defaults to False.
        centralize (bool, optional): whether to centralize gradients, this doesn't work well so don't use it. Defaults to False.
        inner (Chainable | None, optional): preconditioner will be applied to output of this module. Defaults to None.
    """

    def __init__(
        self,
        history_size: int = 10,
        update_freq: int = 1,
        damping: float = 1e-4,
        rdamping: float = 0,
        order: int = 1,
        true_damping: bool = True,
        U_beta: float | None = None,
        S_beta: float | None = None,
        interval: int = 1,
        concat_params: bool = False,
        normalize: bool=False,
        centralize:bool = False,
        inner: Chainable | None = None,
    ):
        # history is still updated each step so Precondition's update_freq has different meaning
        defaults = dict(history_size=history_size, update_freq=update_freq, damping=damping, rdamping=rdamping, true_damping=true_damping, order=order, U_beta=U_beta, S_beta=S_beta, normalize=normalize, centralize=centralize)
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
        normalize = settings['normalize']
        centralize = settings['centralize']

        if 'history' not in state: state['history'] = deque(maxlen=history_size)
        history = state['history']

        if order == 1:
            t = tensor.clone().view(-1)
            if centralize: t -= t.mean()
            if normalize: t /= torch.linalg.vector_norm(t).clip(min=1e-8) # pylint:disable=not-callable
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

                s_k = cur_p - state[f'prev_p_{i}']
                y_k = cur_g - state[f'prev_g_{i}']
                state[f'prev_p_{i}'] = cur_p
                state[f'prev_g_{i}'] = cur_g
                cur_p = s_k
                cur_g = y_k

                if i == order - 1:
                    if centralize: cur_g = cur_g - cur_g.mean()
                    if normalize: cur_g = cur_g / torch.linalg.vector_norm(cur_g).clip(min=1e-8) # pylint:disable=not-callable
                    else: cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=1e-8) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))

        step = state.get('step', 0)
        if step % update_freq == 0 and len(history) != 0:
            U, S_inv = spectral_update(history, damping=damping, rdamping=rdamping, true_damping=true_damping)
            maybe_lerp_(state, U_beta, 'U', U)
            maybe_lerp_(state, S_beta, 'S_inv', S_inv)

        if len(history) != 0:
            state['step'] = step + 1 # do not increment if no history (gathering s_ks and y_ks)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        history_size = settings['history_size']

        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.clip_(-0.1, 0.1) # pyright:ignore[reportArgumentType]

        S_inv = state['S_inv']
        update = spectral_apply(tensor.view(-1), U, S_inv).view_as(tensor)

        n = len(state['history'])
        mh = min(history_size, 10)
        if n <= mh: update.mul_(n/mh)
        return update

