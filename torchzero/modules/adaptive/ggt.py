from collections import deque
from typing import Literal, Any
import warnings

import torch
from ...core import Chainable, TensorTransform
from ...linalg import torch_linalg, regularize_eigh
from .lre_optimizers import LREOptimizerBase

def ggt_update(history: deque[torch.Tensor] | torch.Tensor, damping, rdamping, truncate, eig_tol):
    """returns U ``(ndim, rank)``, L ``(rank, )``"""
    if isinstance(history, torch.Tensor):
        M = history
    else:
        M = torch.stack(tuple(history), dim=1)# / len(history)

    MtM = M.T @ M
    if damping != 0:
        MtM.add_(torch.eye(MtM.size(0), device=MtM.device, dtype=MtM.dtype).mul_(damping))

    try:
        L, Q = torch_linalg.eigh(MtM, retry_float64=True)

        # damping is already added to MTM, rdamping is added afterwards
        L, Q = regularize_eigh(L, Q, truncate=truncate, tol=eig_tol, damping=0, rdamping=0)

        if L is None or Q is None: # this means there are no finite eigenvalues
            return None, None

        U = (M @ Q) * L.rsqrt()

        # this damping is added after computing U, this is why I didn't use one in linalg.regularize_eig
        # that's because we damp singular values this way
        if rdamping != 0:
            L.add_(rdamping * L[-1]) # L is sorted in ascending order

        return L, U

    except torch.linalg.LinAlgError:
        return None, None


class GGT(TensorTransform):
    """
    GGT method from https://arxiv.org/pdf/1806.02958

    The update rule is to stack recent gradients into M, compute U, S <- SVD(M), then calculate update as U S^-1 Uᵀg.
    But it uses eigendecomposition on MᵀM to get U and S^2 because that is faster when you don't neeed V.

    This is equivalent to full-matrix Adagrad on recent gradients.

    Args:
        history_size (int, optional): number of past gradients to store. Defaults to 10.
        beta (float, optional): beta for momentum maintained in whitened space. Defaults to 0.0.
        update_freq (int, optional): frequency of updating the preconditioner (U and S). Defaults to 1.
        eig_tol (float, optional): removes eigenvalues this much smaller than largest eigenvalue. Defaults to 1e-7.
        truncate (int, optional): number of larges eigenvalues to keep. None to disable. Defaults to None.
        damping (float, optional): damping value. Defaults to 1e-4.
        rdamping (float, optional): value of damping relative to largest eigenvalue. Defaults to 0.
        concat_params (bool, optional): if True, treats all parameters as a single vector. Defaults to True.
        inner (Chainable | None, optional): preconditioner will be applied to output of this module. Defaults to None.

    ## Examples:

    Limited-memory Adagrad

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.GGT(),
        tz.m.LR(0.1)
    )
    ```
    Adam with L-Adagrad preconditioner (for debiasing second beta is 0.999 arbitrarily)

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.GGT(inner=tz.m.EMA()),
        tz.m.Debias(0.9, 0.999),
        tz.m.LR(0.01)
    )
    ```

    Stable Adam with L-Adagrad preconditioner (this is what I would recommend)

    ```python
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.GGT(inner=tz.m.EMA()),
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
        update_freq: int = 1,
        eig_tol: float = 1e-7,
        truncate: int | None = None,
        damping: float = 1e-4,
        rdamping: float = 0,
        eigenbasis_optimizer: LREOptimizerBase | None = None,
        concat_params: bool = True,

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults['concat_params']

        super().__init__(defaults, concat_params=concat_params, inner=inner)

    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        history_size = setting['history_size']
        update_freq = setting['update_freq']

        if 'history' not in state: state['history'] = deque(maxlen=history_size)
        history = state['history']

        t = tensor.clone().view(-1)
        history.append(t)

        step = state.get('step', 0)
        state['step'] = step + 1

        if step % update_freq == 0 :

            # compute new factors
            L = state.get("L", None)
            U = state.get("U", None)

            L_new, U_new = ggt_update(
                history,
                damping=setting["damping"],
                rdamping=setting["rdamping"],
                truncate=setting["truncate"],
                eig_tol=setting["eig_tol"],
            )

            # reproject eigenbasis optimizer
            eigenbasis_optimizer: LREOptimizerBase | None = setting["eigenbasis_optimizer"]
            if eigenbasis_optimizer is not None:
                if (L is not None) and (U is not None) and (L_new is not None) and (U_new is not None):
                    eigenbasis_state = state["eigenbasis_state"]
                    eigenbasis_optimizer.reproject(L_old=L, Q_old=U, L_new=L_new, Q_new=U_new, state=eigenbasis_state)


            # store new factors
            if L_new is not None: state["L"] = L_new
            if U_new is not None: state["U"] = U_new


    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        g = tensor.view(-1)
        U = state.get('U', None)

        if U is None:
            # fallback to element-wise preconditioning
            history = torch.stack(tuple(state["history"]), 0)
            g /= history.square().mean(0).sqrt().add(1e-8)
            return g.view_as(tensor)

        L = state['L']

        # step with eigenbasis optimizer
        eigenbasis_optimizer: LREOptimizerBase | None = setting["eigenbasis_optimizer"]
        if eigenbasis_optimizer is not None:

            if "eigenbasis_state" not in state: state["eigenbasis_state"] = {}
            eigenbasis_state = state["eigenbasis_state"]

            update = eigenbasis_optimizer.step(g, L=L, Q=U, state=eigenbasis_state)
            return update.view_as(tensor)

        # or just whiten
        z = U.T @ g
        update = (U * L.rsqrt()) @ z
        return update.view_as(tensor)

