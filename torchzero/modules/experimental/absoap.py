from operator import itemgetter
from typing import Literal

import torch

from ...core import Chainable, Transform
from ..optimizers.shampoo import _merge_small_dims, _unmerge_small_dims
from ..optimizers.soap import project, project_back, get_orthogonal_matrix, get_orthogonal_matrix_QR

@torch.no_grad
def update_absoap_covariances_(
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


Source=Literal['p','g','s','y', 'gy', 'sy', 'sn', 'yn', 'gys', 'sys']
class ABSOAP(Transform):
    """SOAP but with some extra options for testing.

    .. warning::
        This module is just for testing my stupid ideas.

    Args:
        scale_by_s - whether to scale y by s
        gg1 - 1st vector into GGᵀ
        gg2 - 2nd vector into GGᵀ
        ema1 - vector into 1st momentum
        ema2 - 2 vectors into 2nd momentum
        rel1 - if True, multiplies gg1 by params
        rel2 - same but for gg2
        norm - if True, gg1 a and gg2 are normalized, and I need to make that into a letter

    letters:
        p - params
        g - grad
        s - param difference
        y - grad difference
        gy - g+y
        sy - s+y
        sn - s normalized
        yn - y normalized
        gys - g + y#g
        sys - s + y#s

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
        gg1: Source='g',
        gg2: Source='g',
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
            first=gg1,
            second=gg2,
            rel1=rel1, rel2=rel2,
            norm=norm,
        )
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        updates = []
        # update preconditioners
        for i,(p,t, state, setting) in enumerate(zip(params, tensors, states, settings)):
            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps, alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps', 'alpha')(setting)
            scale_by_s = setting['scale_by_s']
            ema1 = setting['ema1']
            ema2 = setting['ema2']
            first=setting['first']
            second=setting['second']
            rel1 = setting['rel1']; rel2 = setting['rel2']
            norm=setting['norm']

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            if 'g_prev' not in state:
                state['p_prev'] = p.clone()
                state['g_prev'] = t.clone()
                # updates.append(tensors[i].clip(-0.1,0.1))
                # continue

            p_prev = state['p_prev']
            g_prev = state['g_prev']
            s = p - p_prev
            y = t - g_prev

            # keep malding
            p_norm = torch.linalg.vector_norm(p) # pylint:disable=not-callable
            g_norm = torch.linalg.vector_norm(t) # pylint:disable=not-callable
            s_norm = torch.linalg.vector_norm(s) # pylint:disable=not-callable
            y_norm = torch.linalg.vector_norm(y) # pylint:disable=not-callable

            sn = p - p_prev * (p_norm / torch.linalg.vector_norm(p_prev))# pylint:disable=not-callable
            yn = t - g_prev * (g_norm / torch.linalg.vector_norm(g_prev))# pylint:disable=not-callable

            if scale_by_s: y /= s_norm.clip(min=1e-8) # pylint:disable=not-callable

            state['p_prev'].copy_(p)
            state['g_prev'].copy_(t)

            def _get(c: Source):
                if c == 'p': return p
                if c == 'g': return t
                if c == 's': return s
                if c == 'y': return y
                if c == 'sn': return sn
                if c == 'yn': return yn
                if c == 'gy': return t+y
                if c == 'sy': return s+y
                if c == 'gys':
                    y_scaled = y * (g_norm/y_norm.clip(min=1e-8))
                    return t+y_scaled
                if c == 'sys':
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
                t1 = t1/torch.linalg.vector_norm(t1).clip(min=1e-8) # pylint:disable=not-callable
                t2 = t2/torch.linalg.vector_norm(t2).clip(min=1e-8) # pylint:disable=not-callable

            # initialize state on 1st step
            if 'GG' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                state["exp_avg_sq"] = torch.zeros_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['GG'] = []

                else:
                    state['GG'] = [torch.zeros(sh, sh, dtype=t.dtype, device=t.device) if 1<sh<max_dim else None for sh in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['GG']]) == 0:
                    state['GG'] = None

                if state['GG'] is not None:
                    update_absoap_covariances_(t1, t2, GGs_=state['GG'], beta=shampoo_beta)
                    state['Q'] = get_orthogonal_matrix(state['GG'])

                state['step'] = 0
                updates.append(tensors[i].clip(-0.1,0.1))
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
                update_absoap_covariances_(t1, t2, state['GG'], shampoo_beta)
                if state['step'] % setting['precond_freq'] == 0:
                    state['Q'], state['exp_avg_sq'] = get_orthogonal_matrix_QR(exp_avg_sq, state['GG'], state['Q'])

        return updates