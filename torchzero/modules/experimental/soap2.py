from operator import itemgetter
import warnings

import torch

from ...core import Chainable, Transform, apply_transform, Module
from ...modules.adaptive.shampoo import _merge_small_dims, _unmerge_small_dims
from ..adaptive.soap import project, project_back, get_orthogonal_matrix, get_orthogonal_matrix_QR
from ...utils import TensorList

@torch.no_grad
def update_soap2_covariances_(
    Hz: torch.Tensor,
    z: torch.Tensor,
    Hs: list[torch.Tensor | None],
    beta: float | None,
):
    for i, H in enumerate(Hs):
        if H is None: continue

        axes = list(range(i)) + list(range(i + 1, z.ndim)) # this works fine with 1d params
        if beta is None: H.add_(torch.tensordot(Hz, z, (axes, axes))) # pyright:ignore[reportArgumentType]

        else: H.lerp_(torch.tensordot(Hz, z, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]


class SOAP2(Module):
    """Soap with Hz z^T
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
        distribution = "rademacher",
        hvp_method = 'autograd',
        h=1e-2,
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
            distribution=distribution,
            hvp_method=hvp_method,
            h=h,
        )
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        z_list = params.sample_like(self.defaults["distribution"])
        Hz_list, _ = var.hessian_vector_product(z_list, rgrad=None, at_x0=True, hvp_method=self.defaults["hvp_method"], h=self.defaults["h"])


        updates = []
        tensors = var.get_update()
        states = [self.state[p] for p in params]
        settings = [self.settings[p] for p in params]

        # update preconditioners
        for i,(p, z, Hz, t, state, setting) in enumerate(zip(params, z_list, Hz_list, tensors, states, settings)):

            beta1, beta2, shampoo_beta, merge_small, max_dim, precondition_1d, eps,alpha = itemgetter(
                'beta1', 'beta2', 'shampoo_beta', 'merge_small', 'max_dim', 'precondition_1d', 'eps','alpha')(setting)

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)
                z, _, _ = _merge_small_dims(z, max_dim)
                Hz, _, _ = _merge_small_dims(Hz, max_dim)

            # initialize state on 1st step
            if 'H' not in state:
                state["exp_avg"] = torch.zeros_like(t)
                state["exp_avg_sq_projected"] = torch.zeros_like(t)

                if not precondition_1d and t.ndim <= 1:
                    state['H'] = []

                else:
                    state['H'] = [torch.zeros(s, s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
                if len([i is not None for i in state['H']]) == 0:
                    state['H'] = None

                if state['H'] is not None:
                    update_soap2_covariances_(Hz, z, Hs=state['H'], beta=shampoo_beta)
                    try: state['Q'] = get_orthogonal_matrix(state['H'])
                    except torch.linalg.LinAlgError as e:
                        warnings.warn(f"torch.linalg.eigh raised an error when initializing SOAP Q matrices on 1st step, diagonal preconditioning will be used for this parameter. The error was:\n{e}")
                        state["H"] = None

                state['step'] = 0
                updates.append(tensors[i].clip(-0.1, 0.1))
                continue  # skip 1st step as in https://github.com/nikhilvyas/SOAP/blob/main/soap.py ?
                # I use scaled update instead as to not mess up with next modules.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            t_projected = None
            if state['H'] is not None:
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
            if state['H'] is not None:
                update_soap2_covariances_(Hz, z, state['H'], shampoo_beta)
                if state['step'] % setting['precond_freq'] == 0:
                    try:
                        state['Q'], state['exp_avg_sq_projected'] = get_orthogonal_matrix_QR(exp_avg_sq_projected, state['H'], state['Q'])
                    except torch.linalg.LinAlgError:
                        pass

        var.update = updates
        return var