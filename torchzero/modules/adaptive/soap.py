import warnings
from operator import itemgetter

import torch

from ...core import Chainable, TensorTransform
from ...linalg import torch_linalg
from ...modules.adaptive.shampoo import _merge_small_dims, _unmerge_small_dims
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states


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
def project(tensor: torch.Tensor, Q: list[torch.Tensor | None]):
    """
    Projects the gradient to the eigenbases of the preconditioner.
    """
    for M in Q:
        if M is not None:
            tensor = torch.tensordot(tensor, M, dims=[[0], [0]]) # pyright:ignore[reportArgumentType]
        else:
            permute_order = list(range(1, len(tensor.shape))) + [0]
            tensor = tensor.permute(permute_order)

    return tensor

@torch.no_grad
def project_back(tensor: torch.Tensor, Q: list[torch.Tensor| None]):
    """
    Projects the gradient back to the original space.
    """
    for M in Q:
        if M is not None:
            tensor = torch.tensordot(tensor, M, dims=[[0], [1]]) # pyright:ignore[reportArgumentType]
        else:
            permute_order = list(range(1, len(tensor.shape))) + [0]
            tensor = tensor.permute(permute_order)

    return tensor

# function from https://github.com/nikhilvyas/SOAP/blob/main/soap.py
@torch.no_grad
def get_orthogonal_matrix(mats: list[torch.Tensor | None]):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """

    final = []
    for M in mats:

        if M is None:
            final.append(None)
            continue

        _, Q = torch_linalg.eigh(M + 1e-30 * torch.eye(M.shape[0], device=M.device), retry_float64=True)

        Q = torch.flip(Q, [1])
        final.append(Q)

    return final

# function from https://github.com/nikhilvyas/SOAP/blob/main/soap.py#L240
@torch.no_grad
def get_orthogonal_matrix_QR(exp_avg_sq: torch.Tensor, GG: list[torch.Tensor | None], Q_list: list[torch.Tensor | None]):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.

    Approximately modifies ``exp_avg_sq`` to be in the new eigenbases.
     """
    final = []

    for ind, (M, O) in enumerate(zip(GG, Q_list)):

        # skip 1d or large dims
        if M is None:
            final.append(None)
            continue

        assert O is not None

        est_eig = torch.diagonal(O.T @ M @ O)
        sort_idx = torch.argsort(est_eig, descending=True)
        exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

        power_iter = M @ O[:, sort_idx]
        Q, _ = torch_linalg.qr(power_iter.to(torch.float32), retry_float64=True)
        Q = Q.to(power_iter.dtype)

        final.append(Q)

    return final, exp_avg_sq

class SOAP(TensorTransform):
    """SOAP (ShampoO with Adam in the Preconditioner's eigenbasis from https://arxiv.org/abs/2409.11321).

    Args:
        beta1 (float, optional): beta for first momentum. Defaults to 0.95.
        beta2 (float, optional): beta for second momentum. Defaults to 0.95.
        shampoo_beta (float | None, optional):
            beta for covariance matrices accumulators. Can be None, then it just sums them like Adagrad (which works worse). Defaults to 0.95.
        precond_freq (int, optional): How often to update the preconditioner. Defaults to 10.
        merge_small (bool, optional): Whether to merge small dims. Defaults to True.
        max_dim (int, optional): Won't precondition dims larger than this. Defaults to 10_000.
        precondition_1d (bool, optional):
            Whether to precondition 1d params (SOAP paper sets this to False). Defaults to True.
        eps (float, optional):
            epsilon for dividing first momentum by second. Defaults to 1e-8.
        debias (bool, optional):
            enables adam bias correction. Defaults to True.
        proj_exp_avg (bool, optional):
            if True, maintains exponential average of gradients (momentum) in projected space.
            If False - in original space Defaults to True.
        alpha (float, optional):
            learning rate. Defaults to 1.
        inner (Chainable | None, optional):
            output of this module is projected and Adam will run on it, but preconditioners are updated
            from original gradients.

    ### Examples:
    SOAP:

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SOAP(),
        tz.m.LR(1e-3)
    )
    ```
    Stabilized SOAP:

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SOAP(),
        tz.m.NormalizeByEMA(max_ema_growth=1.2),
        tz.m.LR(1e-2)
    )
    ```
    """
    def __init__(
        self,
        beta1: float = 0.95,
        beta2: float = 0.95,
        shampoo_beta: float | None = 0.95,
        precond_freq: int = 10,
        merge_small: bool = True,
        max_dim: int = 4096,
        precondition_1d: bool = True,
        eps: float = 1e-8,
        debias: bool = True,
        proj_exp_avg: bool = True,
        alpha: float = 1,

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["inner"]

        super().__init__(defaults)
        self.set_child("inner", inner)

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        if setting["merge_small"]:
            tensor, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(tensor, setting["max_dim"])

        state["exp_avg_proj"] = torch.zeros_like(tensor)
        state["exp_avg_sq_proj"] = torch.zeros_like(tensor)

        if tensor.ndim <= 1 and not setting["precondition_1d"]:
            state['GG'] = []

        else:
            max_dim = setting["max_dim"]
            state['GG'] = [
                torch.zeros(s, s, dtype=tensor.dtype, device=tensor.device) if 1<s<max_dim else None for s in tensor.shape
            ]

        # either scalar parameter, 1d with precondition_1d=False, or all dims are too big.
        if len([i is not None for i in state['GG']]) == 0:
            state['GG'] = None

        # first covariance accumulation
        if state['GG'] is not None:
            update_soap_covariances_(tensor, GGs_=state['GG'], beta=setting["shampoo_beta"])

            # get projection matrix with first gradients with eigh
            try: state['Q'] = get_orthogonal_matrix(state['GG'])
            except torch.linalg.LinAlgError as e:
                warnings.warn(f"torch.linalg.eigh raised an error when initializing SOAP Q matrices on 1st step, diagonal preconditioning will be used for this parameter. The error was:\n{e}")
                state["GG"] = None

        state['step'] = 0


    # no update to avoid running merge_dims twice

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        # note
        # do not modify tensors in-place
        # because they are used to update preconditioner at the end

        steps = [s["step"] for s in states]
        if any(s == 0 for s in steps):
            # skip 1st update so to avoid using current gradient in the projection
            # I scale it instead to avoid issues with further modules
            for s in states: s["step"] += 1
            return TensorList(tensors).clamp(-0.1, 0.1)
            # return TensorList(tensors).zero_()

        fs = settings[0]
        merged_updates = [] # for when exp_avg is maintained unprojected
        merged_grads = [] # this doesn't go into preconditioner
        projected = []

        # -------------------------------- inner step -------------------------------- #
        updates = tensors
        has_inner = "inner" in self.children
        if has_inner:
            updates = self.inner_step_tensors("inner", updates, clone=True,
                                              params=params, grads=grads, loss=loss)

        # ---------------------------------- project --------------------------------- #
        for grad, update, state, setting in zip(tensors, updates, states, settings):
            if setting["merge_small"]:
                update, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(update, setting["max_dim"])
                if has_inner: # grad is a different tensor, merge it too
                    grad, _, _ = _merge_small_dims(grad, setting["max_dim"])
                else: # in this case update is still just grad
                    grad = update

            merged_updates.append(update)
            merged_grads.append(grad)

            if state['GG'] is not None:
                update = project(update, state['Q'])

            projected.append(update)


        # ------------------------ run adam in projected space ----------------------- #
        exp_avg_proj, exp_avg_sq_proj = unpack_states(states, projected, "exp_avg_proj", "exp_avg_sq_proj", must_exist=True, cls=TensorList)
        alpha, beta1, beta2, eps = unpack_dicts(settings, "alpha", "beta1", "beta2", "eps", cls=NumberList)

        # lerp exp_avg in projected space
        if fs["proj_exp_avg"]:
            exp_avg_proj.lerp_(projected, weight=1-beta1)

        # or lerp in original space and project
        else:
            exp_avg = exp_avg_proj
            exp_avg.lerp_(merged_updates, weight=1-beta1)
            exp_avg_proj = []
            for t, state, setting in zip(exp_avg, states, settings):
                if state['GG'] is not None:
                    t = project(t, state["Q"])
                exp_avg_proj.append(t)

        # lerp exp_avg_sq
        exp_avg_sq_proj.mul_(beta2).addcmul_(projected, projected, value=1-beta2)

        # adam direction
        denom = exp_avg_sq_proj.sqrt().add_(eps)
        dirs_proj = exp_avg_proj / denom

        # ------------------------------- project back ------------------------------- #
        dirs: list[torch.Tensor] = []
        for dir, state, setting in zip(dirs_proj, states, settings):
            if state['GG'] is not None:
                dir = project_back(dir, state['Q'])

            if setting["merge_small"]:
                dir = _unmerge_small_dims(dir, state['flat_sizes'], state['sort_idxs'])

            dirs.append(dir)

        # -------------------------- update preconditioners -------------------------- #
        # Update is done after the gradient step to avoid using current gradients in the projection.

        for grad, state, setting in zip(merged_grads, states, settings):
            if state['GG'] is not None:

                # lerp covariances
                update_soap_covariances_(grad, state['GG'], beta=setting["shampoo_beta"])

                # (state['step'] - 1) since we start updating on 2nd step
                if (state['step'] - 1) % setting['precond_freq'] == 0:

                    # unproject exp_avg before updating if it is maintained projected
                    exp_avg = None
                    if fs["proj_exp_avg"]:
                        exp_avg = project_back(state["exp_avg_proj"], state["Q"])

                    # update projection matrix and exp_avg_sq_proj
                    try:
                        state['Q'], state['exp_avg_sq_proj'] = get_orthogonal_matrix_QR(
                            state["exp_avg_sq_proj"], state['GG'], state['Q'])

                        # re-project exp_avg if it is maintained projected
                        if fs["proj_exp_avg"]:
                            assert exp_avg is not None
                            state["exp_avg_proj"] = project(exp_avg, state["Q"])

                    except torch.linalg.LinAlgError:
                        pass

            state["step"] += 1


        # ------------------------- bias-corrected step size ------------------------- #
        if fs["debias"]:
            steps1 = [s+1 for s in steps]
            bias_correction1 = 1.0 - beta1 ** steps1
            bias_correction2 = 1.0 - beta2 ** steps1
            alpha = alpha * (bias_correction2 ** .5) / bias_correction1

        torch._foreach_mul_(dirs, alpha)
        return dirs