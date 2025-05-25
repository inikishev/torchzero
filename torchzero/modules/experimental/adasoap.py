from operator import itemgetter

import torch

from ...core import Chainable, Transform, apply
from ...modules.optimizers.shampoo import _merge_small_dims, _unmerge_small_dims

@torch.no_grad
def update_soap_covariances_(
    grad: torch.Tensor,
    GGs_: list[torch.Tensor | None],
    GG_sqs: list[torch.Tensor | None],
    beta: float | None,
    precond_beta: float | None,
):
    for i, (GG, GG_sq) in enumerate(zip(GGs_, GG_sqs)):
        if GG is None: continue
        assert GG_sq is not None

        if precond_beta is None: GG_sq.addcmul_(GG, GG)
        else: GG_sq.mul_(precond_beta).addcmul_(GG, GG, value=1-precond_beta)

        axes = list(range(i)) + list(range(i + 1, grad.ndim)) # this works fine with 1d params
        if beta is None: GG.add_(torch.tensordot(grad, grad, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: GG.lerp_(torch.tensordot(grad, grad, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

@torch.no_grad
def project(tensors: torch.Tensor, Q: list[torch.Tensor | None]):
    """
    Projects the gradient to the eigenbases of the preconditioner.
    """
    for mat in Q:
        if mat is None: continue
        if len(mat) > 0:
            tensors = torch.tensordot(tensors, mat, dims=[[0], [0]]) # pyright:ignore[reportArgumentType]
        else:
            # I don't understand this part but it is in https://github.com/nikhilvyas/SOAP/blob/main/soap.py
            permute_order = list(range(1, len(tensors.shape))) + [0]
            tensors = tensors.permute(permute_order)

    return tensors

@torch.no_grad
def project_back(tensors: torch.Tensor, Q: list[torch.Tensor| None]):
    """
    Projects the gradient back to the original space.
    """
    for mat in Q:
        if mat is None: continue
        if len(mat) > 0:
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
        if m is None: continue
        if len(m) == 0:
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
        if m is None: continue
        assert o is not None

        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
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

class AdaSOAP(Transform):
    """SOAP with diagonally preconditioned GG^Ts

    precond_beta - beta for GG^T squares
    """
    def __init__(
        self,
        beta1: float = 0.95,
        beta2: float = 0.95,
        shampoo_beta: float | None = 0.95,
        precond_beta: float | None = 0.95,
        precond_freq: int = 10,
        merge_small: bool = True,
        max_dim: int = 2_000,
        precondition_1d: bool = True,
        eps: float = 1e-8,
        decay: float | None = None,
        alpha: float = 1,
        unprojected_exp_avg: bool = True,
        bias_correction: bool = True,
    ):
        defaults = dict(
            beta1=beta1,
            beta2=beta2,
            shampoo_beta=shampoo_beta,
            precond_beta=precond_beta,
            precond_freq=precond_freq,
            merge_small=merge_small,
            max_dim=max_dim,
            precondition_1d=precondition_1d,
            eps=eps,
            decay=decay,
            unprojected_exp_avg=unprojected_exp_avg,
            bias_correction=bias_correction,
            alpha=alpha,
        )
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        updates = []
        # update preconditioners
        for i,(p,t) in enumerate(zip(params, tensors)):
            state = self.state[p]
            settings = self.settings[p]
            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps, unprojected_exp_avg,alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps', 'unprojected_exp_avg','alpha')(settings)
            precond_beta = settings['precond_beta']

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            # initialize state on 1st step
            if 'GG' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                state["exp_avg_sq"] = torch.zeros_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['GG'] = []
                    state['GG_sq'] = []

                else:
                    state['GG'] = [torch.zeros(s, s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]
                    state['GG_sq'] = [torch.zeros(s, s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['GG']]) == 0:
                    state['GG'] = None
                    state['GG_sq'] = None

                if state['GG'] is not None:
                    assert state['GG_sq'] is not None
                    update_soap_covariances_(t, GGs_=state['GG'], GG_sqs=state['GG_sq'], beta=shampoo_beta, precond_beta=precond_beta)
                    GG_precond = [GG / (GG_sq+1e-8) if GG is not None and GG_sq is not None else None for GG, GG_sq in zip(state['GG'], state['GG_sq'])]
                    state['Q'] = get_orthogonal_matrix(GG_precond)

                state['step'] = 0
                updates.append(tensors[i].clip(-0.1,0.1))
                continue  # skip 1st step as in https://github.com/nikhilvyas/SOAP/blob/main/soap.py ?
                # I use sign instead as to not mess up with next modules. 1st Adam step is always sign anyway.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            t_projected = None
            if state['GG'] is not None:
                t_projected = project(t, state['Q'])

            # exponential moving averages
            # this part could be foreached but I will do that at some point its not a big difference compared to preconditioning
            exp_avg: torch.Tensor = state["exp_avg"]
            exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

            if unprojected_exp_avg or t_projected is None:
                exp_avg.lerp_(t, 1-beta1)
            else:
                exp_avg.lerp_(t_projected, 1-beta1)

            if t_projected is None:
                exp_avg_sq.mul_(beta2).addcmul_(t, t, value=1-beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(t_projected, t_projected, value=1-beta2)

            # project exponential moving averages if they are accumulated unprojected
            exp_avg_projected = exp_avg
            if unprojected_exp_avg and t_projected is not None:
                exp_avg_projected = project(exp_avg, state['Q'])

            exp_avg_sq_projected = exp_avg_sq

            denom = exp_avg_sq_projected.sqrt().add_(eps)
            # print(f'{t_projected = }, {exp_avg = }, {exp_avg_projected = }, {exp_avg_sq = }, {exp_avg_sq_projected = }, {denom = }')

            # Projecting back the preconditioned (by Adam) exponential moving average of gradients
            # to the original space
            update = exp_avg_projected / denom
            if t_projected is not None:
                update = project_back(update, state["Q"])

            if settings['bias_correction']:
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
                update_soap_covariances_(t, GGs_=state['GG'], GG_sqs=state['GG_sq'], beta=shampoo_beta, precond_beta=precond_beta)
                GG_precond = [GG / (GG_sq+1e-8) if GG is not None and GG_sq is not None else None for GG, GG_sq in zip(state['GG'], state['GG_sq'])]
                if state['step'] % settings['precond_freq'] == 0:
                    state['Q'], state['exp_avg_sq'] = get_orthogonal_matrix_QR(exp_avg_sq, GG_precond, state['Q'])

        return updates