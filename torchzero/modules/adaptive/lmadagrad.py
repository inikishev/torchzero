from collections import deque
from typing import Literal, Any
import warnings

import torch
from ...core import Chainable, TensorTransform
from ...linalg import torch_linalg

def lm_adagrad_update(history: deque[torch.Tensor] | torch.Tensor, damping, rdamping, truncate, tol):
    """returns U ``(ndim, rank)``, L ``(rank, )``"""
    if isinstance(history, torch.Tensor):
        M = history
    else:
        M = torch.stack(tuple(history), dim=1)# / len(history)

    MTM = M.T @ M
    if damping != 0:
        MTM.add_(torch.eye(MTM.size(0), device=MTM.device, dtype=MTM.dtype).mul_(damping))

    try:
        L, Q = torch_linalg.eigh(MTM, retry_float64=True)

        # remove small eigenvalues relative to largest
        L_max = L.amax()
        indices = L > tol * L_max
        if indices.any():
            L = L[indices]
            Q = Q[:, indices]

        # truncate to top n largest eigenvalues
        if truncate is not None and truncate > 0:
            # L is ordered in ascending order
            L = L[-truncate:]
            Q = Q[:, -truncate:]

        U = (M @ Q) * L.rsqrt()

        if rdamping != 0:
            L.add_(rdamping * L_max)

        return U, L

    except torch.linalg.LinAlgError:
        return None, None

def lm_adagrad_apply(g: torch.Tensor, U: torch.Tensor, L: torch.Tensor, exp_avg_proj: torch.Tensor | None, beta:float):
    z = U.T @ g

    if beta != 0:
        if exp_avg_proj is None: exp_avg_proj = torch.zeros_like(z)
        exp_avg_proj.lerp_(z, weight=1-beta)
        z = exp_avg_proj

    return (U * L.rsqrt()) @ z, exp_avg_proj

def maybe_lerp_(state_: dict, beta: float | None, key, value: Any):
    if value is None: return
    if (key not in state_) or (beta is None): state_[key] = value
    else:
        if state_[key] is None or state_[key].shape != value.shape: state_[key] = value
        else: state_[key].lerp_(value, 1-beta)

class LMAdagrad(TensorTransform):
    """
    Limited-memory full matrix Adagrad.

    The update rule is to stack recent gradients into M, compute U, S <- SVD(M), then calculate update as U S^-1 Uᵀg.
    But it uses eigendecomposition on MᵀM to get U and S^2 because that is faster when you don't neeed V.

    This is equivalent to full-matrix Adagrad on recent gradients.

    Args:
        history_size (int, optional): number of past gradients to store. Defaults to 10.
        beta (float, optional): beta for momentum maintained in whitened space. Defaults to 0.0.
        update_freq (int, optional): frequency of updating the preconditioner (U and S). Defaults to 1.
        damping (float, optional): damping value. Defaults to 1e-4.
        rdamping (float, optional): value of damping relative to singular values norm. Defaults to 0.
        rdamping (float, optional): value of damping relative to singular values norm. Defaults to 0.
        truncate (int, optional): number of larges eigenvalues to keep. None to disable. Defaults to None.
        tol (float, optional): removes eigenvalues this much smaller than largest eigenvalue. Defaults to 1e-7.
        order (int, optional):
            order=2 means gradient differences are used in place of gradients. Higher order uses higher order differences. Defaults to 1.
        U_beta (float | None, optional): momentum for U (too unstable, don't use). Defaults to None.
        L_beta (float | None, optional): momentum for L (too unstable, don't use). Defaults to None.
        concat_params (bool, optional): if True, treats all parameters as a single vector. Defaults to True.
        inner (Chainable | None, optional): preconditioner will be applied to output of this module. Defaults to None.

    ## Examples:

    Limited-memory Adagrad

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LMAdagrad(),
        tz.m.LR(0.1)
    )
    ```
    Adam with L-Adagrad preconditioner (for debiasing second beta is 0.999 arbitrarily)

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LMAdagrad(inner=tz.m.EMA()),
        tz.m.Debias(0.9, 0.999),
        tz.m.LR(0.01)
    )
    ```

    Stable Adam with L-Adagrad preconditioner (this is what I would recommend)

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LMAdagrad(inner=tz.m.EMA()),
        tz.m.Debias(0.9, 0.999),
        tz.m.ClipNormByEMA(max_ema_growth=1.2),
        tz.m.LR(0.01)
    )
    ```
    Reference:
        Agarwal N. et al. Efficient full-matrix adaptive regularization //International Conference on Machine Learning. – PMLR, 2019. – С. 102-110.
    """

    def __init__(
        self,
        history_size: int = 100,
        beta: float = 0.0,
        update_freq: int = 1,
        damping: float = 1e-4,
        rdamping: float = 0,
        truncate: int | None = None,
        tol: float = 1e-7,
        order: int = 1,
        U_beta: float | None = None,
        L_beta: float | None = None,
        concat_params: bool = True,

        inner: Chainable | None = None,
        U_tfm: Chainable | None = None,
        L_tfm: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults['concat_params'], defaults["U_tfm"], defaults["L_tfm"]

        super().__init__(defaults, concat_params=concat_params, inner=inner)

        self.set_child("U", U_tfm)
        self.set_child("L", L_tfm)


    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        order = setting['order']
        history_size = setting['history_size']
        update_freq = setting['update_freq']
        U_beta = setting['U_beta']
        L_beta = setting['L_beta']

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
            eps = torch.finfo(cur_p.dtype).tiny * 2
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
                    cur_g = cur_g / torch.linalg.norm(cur_p).clip(min=eps) # pylint:disable=not-callable
                    history.append(cur_g.view(-1))

        step = state.get('step', 0)
        if step % update_freq == 0 and len(history) != 0:

            # if maintaining momentum, unproject exp_avg before updating factors and reproject
            exp_avg_proj = state.get("exp_avg_proj", None)
            exp_avg = None
            if exp_avg_proj is not None and "U" in state:
                exp_avg = state["U"] @ exp_avg_proj

            # update factors
            U, L = lm_adagrad_update(
                history,
                damping=setting["damping"],
                rdamping=setting["rdamping"],
                truncate=setting["truncate"],
                tol=setting["tol"],
            )
            maybe_lerp_(state, U_beta, 'U', U)
            maybe_lerp_(state, L_beta, 'L', L)

            # re-project exp_avg with new factors
            if U is not None and exp_avg_proj is not None:
                assert exp_avg is not None
                state["exp_avg_proj"] = U.T @ exp_avg


        if len(history) != 0:
            state['step'] = step + 1 # do not increment if no history (gathering s_ks and y_ks)

    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        U = state.get('U', None)
        if U is None:
            # make a conservative step to avoid issues due to different GD scaling
            return tensor.clip_(-0.1, 0.1)

        # -------------------------------- transforms -------------------------------- #
        L = state['L']
        if "L" in self.children:
            if not self._concat_params: raise RuntimeError("L/U transforms can only be used with concat_params=True")
            L = self.inner_step_tensors("L", [L], clone=True)[0]

        if "U" in self.children:
            if not self._concat_params: raise RuntimeError("L/U transforms can only be used with concat_params=True")
            U = self.inner_step_tensors("U", [U], clone=True)[0]

        # ------------------------------- precondition ------------------------------- #
        g = tensor.view(-1)
        exp_avg_proj = state.get("exp_avg_proj", None)
        update, state["exp_avg_proj"] = lm_adagrad_apply(g, U, L, exp_avg_proj, beta=setting["beta"])
        return update.view_as(tensor)

