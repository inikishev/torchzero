from operator import itemgetter
import warnings

import torch

from ...core import TensorTransform, Chainable, Module
from ..adaptive import Adam
from ...utils import unpack_dicts, unpack_states, TensorList, NumberList, set_storage_
from ...modules.adaptive.shampoo import _merge_small_dims, _unmerge_small_dims
from ...linalg import torch_linalg
from ..adaptive.soap import get_orthogonal_matrix, project, project_back, update_soap_covariances_

# function from https://github.com/nikhilvyas/SOAP/blob/main/soap.py#L240
@torch.no_grad
def get_orthogonal_matrix_QR(grad_sqs: list[torch.Tensor], GG: list[torch.Tensor | None], Q_list: list[torch.Tensor | None]):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
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
        grad_sqs = [s.index_select(ind, sort_idx) for s in grad_sqs]

        power_iter = M @ O[:, sort_idx]
        Q, _ = torch_linalg.qr(power_iter.to(torch.float32), retry_float64=True)
        Q = Q.to(power_iter.dtype)

        final.append(Q)

    return final, grad_sqs

class SOAPBasis(TensorTransform):
    """
    Run another optimizer in Shampoo eigenbases.

    Note:
        the buffers of the ``basis_opt`` are re-projected whenever basis changes. The reprojection logic is not implemented on all modules. Some supported modules are:

        ``Adagrad``, ``Adam``, ``Adan``, ``Lion``, ``MARSCorrection``, ``MSAMMomentum``, ``RMSprop``, ``EMA``, ``HeavyBall``, ``NAG``, ``ClipNormByEMA``, ``ClipValueByEMA``, ``NormalizeByEMA``, ``ClipValueGrowth``, ``CoordinateMomentum``, ``CubicAdam``.

        Additionally most modules with no internal buffers are supported, e.g. ``Cautious``, ``Sign``, ``ClipNorm``, ``Orthogonalize``, etc. However modules that use weight values, such as ``WeighDecay`` can't be supported, as weights can't be projected.

        Also, if you say use ``EMA`` on output of ``Pow(2)``, the exponential average will be reprojected as gradient and not as squared gradients. Use modules like ``EMASquared``, ``SqrtEMASquared`` to get correct reprojections.

    Args:
        basis_opt (Chainable): module or modules to run in Shampoo eigenbases.
        shampoo_beta (float | None, optional):
            beta for covariance matrices accumulators. Can be None, then it just sums them like Adagrad (which works worse). Defaults to 0.95.
        precond_freq (int, optional): How often to update the preconditioner. Defaults to 10.
        merge_small (bool, optional): Whether to merge small dims. Defaults to True.
        max_dim (int, optional): Won't precondition dims larger than this. Defaults to 10_000.
        precondition_1d (bool, optional):
            Whether to precondition 1d params (SOAP paper sets this to False). Defaults to True.
        inner (Chainable | None, optional):
            output of this module is projected and ``basis_opt`` will run on it, but preconditioners are updated
            from original gradients.

    Examples:
    SOAP with MARS and AMSGrad:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SOAPBasis([tz.m.MARSCorrection(0.95), tz.m.Adam(0.95, 0.95, amsgrad=True)]),
        tz.m.LR(1e-3)
    )
    ```

    LaProp in Shampoo eigenbases (SOLP):
    ```python

    # we define LaProp through other modules, moved it out for brevity
    laprop = (
        tz.m.RMSprop(0.95),
        tz.m.Debias(beta1=None, beta2=0.95),
        tz.m.EMA(0.95),
        tz.m.Debias(beta1=0.95, beta2=None),
    )

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SOAPBasis(laprop),
        tz.m.LR(1e-3)
    )
    ```

    Lion in Shampoo eigenbases (works kinda well):
    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SOAPBasis(tz.m.Lion()),
        tz.m.LR(1e-3)
    )
    ```
    """
    def __init__(
        self,
        basis_opt: Chainable,
        shampoo_beta: float | None = 0.95,
        precond_freq: int = 10,
        merge_small: bool = True,
        max_dim: int = 4096,
        precondition_1d: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["inner"], defaults["basis_opt"]

        super().__init__(defaults)
        self.set_child("inner", inner)
        self.set_child("basis_opt", basis_opt)

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


        # ------------------------ run opt in projected space ----------------------- #
        dirs_proj = self.inner_step_tensors("basis_opt", tensors=projected, clone=True, grads=projected)

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

        grad_buffs = self.get_child_projected_buffers("basis_opt", "grad")
        grad_sq_buffs = self.get_child_projected_buffers("basis_opt", ["grad_sq", "grad_cu"])

        for i, (grad, state, setting) in enumerate(zip(merged_grads, states, settings)):
            if state['GG'] is not None:

                # lerp covariances
                update_soap_covariances_(grad, state['GG'], beta=setting["shampoo_beta"])

                # (state['step'] - 1) since we start updating on 2nd step
                if (state['step'] - 1) % setting['precond_freq'] == 0:
                    g_buffs = [b[i] for b in grad_buffs]
                    g_sq_buffs = [b[i] for b in grad_sq_buffs]

                    # unproject grad buffers before updating
                    g_buffs_unproj = [project_back(buff, state["Q"]) for buff in g_buffs]

                    # update projection matrix and exp_avg_sq_proj
                    try:
                        state['Q'], g_sq_buffs_new = get_orthogonal_matrix_QR(
                            g_sq_buffs, state['GG'], state['Q'])

                        for b_old, b_new in zip(g_sq_buffs, g_sq_buffs_new):
                            set_storage_(b_old, b_new)

                        # re-project grad buffers
                        for b_proj, b_unproj in zip(g_buffs, g_buffs_unproj):
                            set_storage_(b_proj, project(b_unproj, state["Q"]))

                    except torch.linalg.LinAlgError:
                        pass

            state["step"] += 1

        return dirs