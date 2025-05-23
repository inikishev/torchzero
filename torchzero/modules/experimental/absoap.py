from operator import itemgetter

import torch
from typing import Literal
from ...core import Chainable, Transform, apply
from ..optimizers.shampoo import _merge_small_dims, _unmerge_small_dims

@torch.no_grad
def update_soap_covariances_(
    g1: torch.Tensor,
    g2: torch.Tensor,
    GGs_: list[torch.Tensor | None],
    beta: float | None,
):
    for i, GG in enumerate(GGs_):
        if GG is None: continue

        axes = list(range(i)) + list(range(i + 1, g1.ndim)) # this works fine with 1d params
        if beta is None: GG.add_(torch.tensordot(g1, g2, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: GG.lerp_(torch.tensordot(g1, g2, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

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

Source=Literal['p','g','s','y', 'gy', 'sy', 'gys', 'sys']
class ABSOAP(Transform):
    """SOAP but with two extra letters included in its name in order to improve converence

    new args

    scale by s whether to scale gradient differences by parameter differences

    y_to_ema2 whether to use gradient differences for exponential moving average too
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
        scale_by_s: bool = True,
        first: Source='g',
        second: Source='g',
        ema1: Source='g',
        ema2: tuple[Source, Source] = ('g','g'),
        rel1: bool=False,
        rel2: bool=False,
        norm: bool = False,
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
            scale_by_s=scale_by_s,
            ema1=ema1,
            ema2=ema2,
            first=first,
            second=second,
            rel1=rel1, rel2=rel2,
            norm=norm,
        )
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        updates = []
        # update preconditioners
        for i,(p,t) in enumerate(zip(params, tensors)):
            state = self.state[p]
            settings = self.settings[p]
            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps, alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps', 'alpha')(settings)
            scale_by_s = settings['scale_by_s']
            ema1 = settings['ema1']
            ema2 = settings['ema2']
            first=settings['first']
            second=settings['second']
            rel1 = settings['rel1']; rel2 = settings['rel2']
            norm=settings['norm']

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            if 'g_prev' not in state:
                state['p_prev'] = p.clone()
                state['g_prev'] = t.clone()
                updates.append(tensors[i].sign())
                continue

            p_prev = state['p_prev']
            g_prev = state['g_prev']
            s = p - p_prev
            y = t - g_prev
            if scale_by_s: y /= torch.linalg.norm(s).clip(min=1e-8) # pylint:disable=not-callable

            state['p_prev'].copy_(p)
            state['g_prev'].copy_(t)

            def _get(c: Source):
                if c == 'p': return p
                if c == 'g': return t
                if c == 's': return s
                if c == 'y': return y
                if c == 'gy': return t+y
                if c == 'sy': return s+y
                if c == 'gys':
                    g_norm = torch.linalg.norm(t) # pylint:disable=not-callable
                    y_norm = torch.linalg.norm(y) # pylint:disable=not-callable
                    y_scaled = y * (g_norm/y_norm.clip(min=1e-8))
                    return t+y_scaled
                if c == 'sys':
                    s_norm = torch.linalg.norm(s) # pylint:disable=not-callable
                    y_norm = torch.linalg.norm(y) # pylint:disable=not-callable
                    y_scaled = y * (s_norm/y_norm.clip(min=1e-8))
                    return s+y_scaled
                raise RuntimeError("Big Chungus")

            t1 = _get(first)
            if rel1: t1 = t1 * p.abs().clip(min=1e-6)
            t2 = _get(second)
            if rel2: t2 = t2 * p.abs().clip(min=1e-6)

            t_ema1 = _get(ema1)
            t_ema2s = _get(ema2[0]), _get(ema2[1])

            if norm:
                t1 = t1/torch.linalg.norm(t1).clip(min=1e-8) # pylint:disable=not-callable
                t2 = t2/torch.linalg.norm(t2).clip(min=1e-8) # pylint:disable=not-callable


            # initialize state on 1st step
            if 'GG' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                state["exp_avg_sq"] = torch.ones_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['GG'] = []

                else:
                    state['GG'] = [torch.zeros(sh, sh, dtype=t.dtype, device=t.device) if 1<sh<max_dim else None for sh in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['GG']]) == 0:
                    state['GG'] = None

                if state['GG'] is not None:
                    update_soap_covariances_(t1, t2, GGs_=state['GG'], beta=shampoo_beta)
                    state['Q'] = get_orthogonal_matrix(state['GG'])

                state['step'] = 0
                updates.append(tensors[i].sign())
                continue  # skip 1st step as in https://github.com/nikhilvyas/SOAP/blob/main/soap.py ?
                # I use sign instead as to not mess up with next modules. 1st Adam step is always sign anyway.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            z1_projected = None
            z2_projected = None

            if state['GG'] is not None:
                z1_projected = project(t_ema2s[0], state['Q'])
                if ema2[0] == ema2[1]: z2_projected = z1_projected
                else: z2_projected = project(t_ema2s[1], state['Q'])

            # exponential moving averages
            # this part could be foreached but I will do that at some point its not a big difference compared to preconditioning
            exp_avg: torch.Tensor = state["exp_avg"]
            exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

            exp_avg.lerp_(t_ema1, 1-beta1)

            if z1_projected is None:
                exp_avg_sq.mul_(beta2).addcmul_(*t_ema2s, value=1-beta2)
            else:
                assert z2_projected is not None
                exp_avg_sq.mul_(beta2).addcmul_(z1_projected, z2_projected, value=1-beta2)

            # project exponential moving averages if they are accumulated unprojected
            exp_avg_projected = exp_avg
            if z1_projected is not None:
                exp_avg_projected = project(exp_avg, state['Q'])

            exp_avg_sq_projected = exp_avg_sq

            denom = exp_avg_sq_projected.sqrt().add_(eps)
            # print(f'{t_projected = }, {exp_avg = }, {exp_avg_projected = }, {exp_avg_sq = }, {exp_avg_sq_projected = }, {denom = }')

            # Projecting back the preconditioned (by Adam) exponential moving average of gradients
            # to the original space
            update = exp_avg_projected / denom
            if z1_projected is not None:
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
                update_soap_covariances_(t1, t2, state['GG'], shampoo_beta)
                if state['step'] % settings['precond_freq'] == 0:
                    state['Q'], state['exp_avg_sq'] = get_orthogonal_matrix_QR(exp_avg_sq, state['GG'], state['Q'])

        return updates