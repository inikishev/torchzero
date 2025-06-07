from operator import itemgetter

import torch

from ...core import Chainable, Transform
from ..optimizers.shampoo import _merge_small_dims, _unmerge_small_dims
from ..optimizers.soap import (
    update_soap_covariances_,
    get_orthogonal_matrix,
    get_orthogonal_matrix_QR,
    project,
    project_back,
)

class SOAPY(Transform):
    """Adam but uses scaled gradient differences for GGᵀ. Please note that this is experimental and isn't guaranteed to work.

    New args:
        scale_by_s - whether to scale gradient differences by parameter differences
        y_to_ema2 - whether to use gradient differences for exponential moving average too
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
        y_to_ema2: bool = False,
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
            y_to_ema2=y_to_ema2,
        )
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply(self, tensors, params, grads, loss, states, settings):
        updates = []
        # update preconditioners
        for i,(p,t, state, setting) in enumerate(zip(params, tensors, states, settings)):
            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps, alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps', 'alpha')(setting)
            scale_by_s = setting['scale_by_s']
            y_to_ema2 = setting['y_to_ema2']

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            if 'g_prev' not in state:
                state['p_prev'] = p.clone()
                state['g_prev'] = t.clone()
                updates.append(tensors[i].clip(-0.1,0.1))
                continue

            p_prev = state['p_prev']
            g_prev = state['g_prev']
            s = p - p_prev
            y = t - g_prev
            if scale_by_s: y /= torch.linalg.norm(s).clip(min=1e-8) # pylint:disable=not-callable

            state['p_prev'].copy_(p)
            state['g_prev'].copy_(t)

            # initialize state on 1st step
            if 'GG' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                if y_to_ema2: state["exp_avg_sq"] = torch.ones_like(t)
                else: state["exp_avg_sq"] = torch.zeros_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['GG'] = []

                else:
                    state['GG'] = [torch.zeros(sh, sh, dtype=t.dtype, device=t.device) if 1<sh<max_dim else None for sh in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['GG']]) == 0:
                    state['GG'] = None

                if state['GG'] is not None:
                    update_soap_covariances_(y, GGs_=state['GG'], beta=shampoo_beta)
                    state['Q'] = get_orthogonal_matrix(state['GG'])

                state['step'] = 0
                updates.append(tensors[i].clip(-0.1,0.1))
                continue  # skip 1st step as in https://github.com/nikhilvyas/SOAP/blob/main/soap.py ?
                # I use sign instead as to not mess up with next modules. 1st Adam step is always sign anyway.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            z_projected = None
            if state['GG'] is not None:
                if y_to_ema2: z_projected = project(y, state['Q'])
                else: z_projected = project(t, state['Q'])

            # exponential moving averages
            # this part could be foreached but I will do that at some point its not a big difference compared to preconditioning
            exp_avg: torch.Tensor = state["exp_avg"]
            exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

            exp_avg.lerp_(t, 1-beta1)

            if z_projected is None:
                if y_to_ema2: exp_avg_sq.mul_(beta2).addcmul_(y, y, value=1-beta2)
                else: exp_avg_sq.mul_(beta2).addcmul_(t, t, value=1-beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(z_projected, z_projected, value=1-beta2)

            # project exponential moving averages if they are accumulated unprojected
            exp_avg_projected = exp_avg
            if z_projected is not None:
                exp_avg_projected = project(exp_avg, state['Q'])

            exp_avg_sq_projected = exp_avg_sq

            denom = exp_avg_sq_projected.sqrt().add_(eps)
            # print(f'{t_projected = }, {exp_avg = }, {exp_avg_projected = }, {exp_avg_sq = }, {exp_avg_sq_projected = }, {denom = }')

            # Projecting back the preconditioned (by Adam) exponential moving average of gradients
            # to the original space
            update = exp_avg_projected / denom
            if z_projected is not None:
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
                update_soap_covariances_(y, state['GG'], shampoo_beta)
                if state['step'] % setting['precond_freq'] == 0:
                    state['Q'], state['exp_avg_sq'] = get_orthogonal_matrix_QR(exp_avg_sq, state['GG'], state['Q'])

        return updates