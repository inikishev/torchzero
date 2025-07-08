from operator import itemgetter

import torch

from ...core import Chainable, Transform, apply_transform
from ...modules.optimizers.shampoo import _merge_small_dims, _unmerge_small_dims

@torch.no_grad
def update_soap_covariances_(
    grad: torch.Tensor,
    GGs_: list[torch.Tensor | None],
    beta: float | None,
):
    for i, GG in enumerate(GGs_):
        if GG is None: continue

        axes = list(range(i)) + list(range(i + 1, grad.ndim)) # this works fine with 1d params
        if beta is None: GG.add_(torch.tensordot(grad, grad, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: GG.lerp_(torch.tensordot(grad, grad, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

@torch.no_grad
def project(tensors: torch.Tensor, Q: list[torch.Tensor | None]):
    """
    Projects the gradient to the eigenbases of the preconditioner.
    """
    for mat in Q:
        if mat is not None and len(mat) > 0:
            tensors = torch.tensordot(tensors, mat, dims=[[0], [0]]) # pyright:ignore[reportArgumentType]
        else:
            permute_order = list(range(1, len(tensors.shape))) + [0]
            tensors = tensors.permute(permute_order)

    return tensors

@torch.no_grad
def project_back(tensors: torch.Tensor, Q: list[torch.Tensor| None]):
    """
    Projects the gradient back to the original space.
    """
    for mat in Q:
        if mat is not None and len(mat) > 0:
            tensors = torch.tensordot(tensors, mat,dims=[[0], [1]]) # pyright:ignore[reportArgumentType]
        else:
            permute_order = list(range(1, len(tensors.shape))) + [0]
            tensors = tensors.permute(permute_order)

    return tensors

# function from https://github.com/nikhilvyas/SOAP/blob/main/soap.py
@torch.no_grad
def get_orthogonal_matrix(mat: list[torch.Tensor | None]):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    float_data = False
    original_type = original_device = None
    for m in mat:
        if m is None or len(m) == 0:
            matrix.append([])
            continue
        if m.dtype != torch.float:
            original_type = m.dtype
            original_device = m.device
            matrix.append(m.float())
        else:
            float_data = True
            matrix.append(m)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue
        try:
            _, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device)) # pylint:disable=not-callable
        except Exception:
            _, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device)) # pylint:disable=not-callable
            Q = Q.to(m.dtype)
        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)
    return final

# function from https://github.com/nikhilvyas/SOAP/blob/main/soap.py#L240
@torch.no_grad
def get_orthogonal_matrix_QR(exp_avg_sq: torch.Tensor, GG: list[torch.Tensor | None], Q_list: list[torch.Tensor | None]):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    float_data = False
    original_type = original_device = None
    for m,o in zip(GG, Q_list):
        if m is None or len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        assert o is not None
        if m.data.dtype != torch.float:
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())
        else:
            float_data = True
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

    final = []
    for ind, (m,o) in enumerate(zip(matrix, orth_matrix)):
        if len(m)==0:
            final.append([])
            continue
        est_eig = torch.diag(o.T @ m @ o)
        sort_idx = torch.argsort(est_eig, descending=True)
        exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
        o = o[:,sort_idx]
        power_iter = m @ o
        Q, _ = torch.linalg.qr(power_iter) # pylint:disable=not-callable

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    return final, exp_avg_sq

class SOAP(Transform):
    """SOAP (ShampoO with Adam in the Preconditioner's eigenbasis from https://arxiv.org/abs/2409.11321).

    Args:
        beta1 (float, optional): beta for first momentum. Defaults to 0.95.
        beta2 (float, optional): beta for second momentum. Defaults to 0.95.
        shampoo_beta (float | None, optional):
            beta for covariance matrices accumulators. Can be None, then it just sums them like Adagrad (which works worse). Defaults to 0.95.
        precond_freq (int, optional): How often to update the preconditioner. Defaults to 10.
        merge_small (bool, optional): Whether to merge small dims. Defaults to True.
        max_dim (int, optional): Won't precondition dims larger than this. Defaults to 2_000.
        precondition_1d (bool, optional):
            Whether to precondition 1d params (SOAP paper sets this to False). Defaults to True.
        eps (float, optional):
            epsilon for dividing first momentum by second. Defaults to 1e-8.
        decay (float | None, optional):
            Decays covariance matrix accumulators, this may be useful if `shampoo_beta` is None. Defaults to None.
        alpha (float, optional):
            learning rate. Defaults to 1.
        bias_correction (bool, optional):
            enables adam bias correction. Defaults to True.

    Examples:
        SOAP:

        .. code-block:: python

            opt = tz.Modular(model.parameters(), tz.m.SOAP(), tz.m.LR(1e-3))

        Stabilized SOAP:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SOAP(),
                tz.m.NormalizeByEMA(max_ema_growth=1.2),
                tz.m.LR(1e-2)
            )
    """
    def __init__(
        self,
        beta1: float = 0.95,
        beta2: float = 0.95,
        shampoo_beta: float | None = 0.95,
        precond_freq: int = 10,
        merge_small: bool = True,
        max_dim: int = 2_000,
        precondition_1d: bool = True,
        eps: float = 1e-8,
        decay: float | None = None,
        alpha: float = 1,
        bias_correction: bool = True,
    ):
        defaults = dict(
            beta1=beta1,
            beta2=beta2,
            shampoo_beta=shampoo_beta,
            precond_freq=precond_freq,
            merge_small=merge_small,
            max_dim=max_dim,
            precondition_1d=precondition_1d,
            eps=eps,
            decay=decay,
            bias_correction=bias_correction,
            alpha=alpha,
        )
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        updates = []
        # update preconditioners
        for i,(p,t, state, setting) in enumerate(zip(params, tensors, states, settings)):
            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps,alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps','alpha')(setting)

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            # initialize state on 1st step
            if 'GG' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                state["exp_avg_sq_projected"] = torch.zeros_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['GG'] = []

                else:
                    state['GG'] = [torch.zeros(s, s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['GG']]) == 0:
                    state['GG'] = None

                if state['GG'] is not None:
                    update_soap_covariances_(t, GGs_=state['GG'], beta=shampoo_beta)
                    state['Q'] = get_orthogonal_matrix(state['GG'])

                state['step'] = 0
                updates.append(tensors[i].clip(-0.1, 0.1))
                continue  # skip 1st step as in https://github.com/nikhilvyas/SOAP/blob/main/soap.py ?
                # I use scaled update instead as to not mess up with next modules.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            t_projected = None
            if state['GG'] is not None:
                t_projected = project(t, state['Q'])

            # exponential moving averages
            # this part could be foreached but I will do that at some point its not a big difference compared to preconditioning
            exp_avg: torch.Tensor = state["exp_avg"]
            exp_avg_sq_projected: torch.Tensor = state["exp_avg_sq_projected"]

            exp_avg.lerp_(t, 1-beta1)

            if t_projected is None:
                exp_avg_sq_projected.mul_(beta2).addcmul_(t, t, value=1-beta2)
            else:
                exp_avg_sq_projected.mul_(beta2).addcmul_(t_projected, t_projected, value=1-beta2)

            # project exponential moving averages if they are accumulated unprojected
            exp_avg_projected = exp_avg
            if t_projected is not None:
                exp_avg_projected = project(exp_avg, state['Q'])

            denom = exp_avg_sq_projected.sqrt().add_(eps)
            # print(f'{t_projected = }, {exp_avg = }, {exp_avg_projected = }, {exp_avg_sq = }, {exp_avg_sq_projected = }, {denom = }')

            # Projecting back the preconditioned (by Adam) exponential moving average of gradients
            # to the original space
            update = exp_avg_projected / denom

            if t_projected is not None:
                update = project_back(update, state["Q"])

            if setting['bias_correction']:
                bias_correction1 = 1.0 - beta1 ** (state["step"]+1)
                bias_correction2 = 1.0 - beta2 ** (state["step"]+1)
                update *= ((bias_correction2 ** .5) / bias_correction1) * alpha
            elif alpha is not None:
                update *= alpha

            if merge_small:
                update = _unmerge_small_dims(update, state['flat_sizes'], state['sort_idxs'])

            updates.append(update)
            state["step"] += 1

            # Update is done after the gradient step to avoid using current gradients in the projection.
            if state['GG'] is not None:
                update_soap_covariances_(t, state['GG'], shampoo_beta)
                if state['step'] % setting['precond_freq'] == 0:
                    state['Q'], state['exp_avg_sq_projected'] = get_orthogonal_matrix_QR(exp_avg_sq_projected, state['GG'], state['Q'])

        return updates