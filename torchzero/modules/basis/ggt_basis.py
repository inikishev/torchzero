from collections import deque

import torch

from ...core import Chainable, TensorTransform
from ...utils import set_storage_
from ..adaptive.ggt import ggt_update


def _cubic_reproject(C: torch.Tensor, cu: torch.Tensor, approx:bool):
    if approx: return C.pow(3) @ cu

    n = cu.numel()
    T = torch.zeros([n,n,n], device=cu.device, dtype=cu.dtype)
    T[range(n),range(n),range(n)] = cu
    T = torch.einsum('ai,bj,ck,ijk->abc', C, C, C, T)
    n2 = T.size(0)
    return T[range(n2), range(n2), range(n2)]

class GGTBasis(TensorTransform):
    """
    Run another optimizer in GGT eigenbasis. The eigenbasis is ``rank``-sized, so it is possible to run expensive
    methods such as Full-matrix Adagrad/Adam.

    The update rule is to stack recent gradients into M and
    compute eigendecomposition of M M^T via eigendecomposition of M^T M.

    This is equivalent to full-matrix Adagrad on recent gradients.

    Note:
        the buffers of the ``basis_opt`` are re-projected whenever basis changes. The reprojection logic is not implemented on all modules. Some supported modules are:

        ``Adagrad``, ``FullMatrixAdagrad``, ``Adam``, ``Adan``, ``Lion``, ``MARSCorrection``, ``MSAMMomentum``, ``RMSprop``, ``GGT``, ``EMA``, ``HeavyBall``, ``NAG``, ``ClipNormByEMA``, ``ClipValueByEMA``, ``NormalizeByEMA``, ``ClipValueGrowth``, ``CoordinateMomentum``, ``CubicAdam``.

        Additionally most modules with no internal buffers are supported, e.g. ``Cautious``, ``Sign``, ``ClipNorm``, ``Orthogonalize``, etc. However modules that use weight values, such as ``WeighDecay`` can't be supported, as weights can't be projected.

        Also, if you say use ``EMA`` on output of ``Pow(2)``, the exponential average will be reprojected as gradient and not as squared gradients. Use modules like ``EMASquared``, ``SqrtEMASquared`` to get correct reprojections.


    Args:
        basis_opt (Chainable): module or modules to run in GGT eigenbasis.
        history_size (int, optional): number of past gradients to store, and rank of preconditioner. Defaults to 10.
        update_freq (int, optional): frequency of updating the preconditioner (U and S). Defaults to 1.
        eig_tol (float, optional): removes eigenvalues this much smaller than largest eigenvalue. Defaults to 1e-7.
        truncate (int, optional): number of larges eigenvalues to keep. None to disable. Defaults to None.
        damping (float, optional): damping value. Defaults to 1e-4.
        rdamping (float, optional): value of damping relative to largest eigenvalue. Defaults to 0.
        concat_params (bool, optional): if True, treats all parameters as a single vector. Defaults to True.
        inner (Chainable | None, optional):
            output of this module is projected and ``basis_opt`` will run on it, but preconditioners are updated
            from original gradients.

    ## Examples:

    Examples:
    Adam in GGT eigenbasis:
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.GGTBasis(tz.m.Adam(beta2=0.99)),
        tz.m.LR(1e-3)
    )
    ```

    Full-matrix Adam in GGT eigenbasis. We can define full-matrix Adam through ``FullMatrixAdagrad``.
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.GGTBasis(
            [tz.m.FullMatrixAdagrad(beta=0.99, inner=tz.m.EMA(0.9, debias=True))]
        ),
        tz.m.LR(1e-3)
    )
    ```

    LaProp in GGT eigenbasis:
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
        tz.m.GGTBasis(laprop),
        tz.m.LR(1e-3)
    )
    ```

    Reference:
        Agarwal N. et al. Efficient full-matrix adaptive regularization //International Conference on Machine Learning. – PMLR, 2019. – С. 102-110.
    """

    def __init__(
        self,
        basis_opt: Chainable,
        history_size: int = 100,
        update_freq: int = 1,
        eig_tol: float = 1e-7,
        truncate: int | None = None,
        damping: float = 1e-4,
        rdamping: float = 0,
        matrix_power: float = -1/2,
        approx_sq_reproject:bool = False,
        approx_cu_reproject:bool = False,

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']

        super().__init__(defaults, concat_params=True, inner=inner)
        self.set_child("basis_opt", basis_opt)

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
                matrix_power=setting["matrix_power"],
            )

            if (L is not None) and (U is not None) and (L_new is not None) and (U_new is not None):
                # reproject basis optimizer
                # this happens after first step, so basis opt is initialized by then
                # note that because we concatenate parameters, each buffer will a single rank-length vector
                C = U_new.T @ U # change of basis matrix

                # reproject gradient-like buffers
                for (buff,) in self.get_child_projected_buffers("basis_opt", "grad"):
                    set_storage_(buff, C @ buff)

                # reproject covariance diagonal-like buffers
                for (buff,) in self.get_child_projected_buffers("basis_opt", "grad_sq"):
                    if setting["approx_sq_reproject"]: set_storage_(buff, C.pow(2) @ buff)
                    else: set_storage_(buff, (C @ buff.diag_embed() @ C.T).diagonal())

                # reproject third order diagonal-like buffers
                for (buff,) in self.get_child_projected_buffers("basis_opt", "grad_cu"):
                    buff_r = _cubic_reproject(C, buff, setting["approx_cu_reproject"])
                    set_storage_(buff, buff_r)

                # reproject covariance-like buffers
                for (buff,) in self.get_child_projected_buffers("basis_opt", "covariance"):
                    set_storage_(buff, C @ buff @ C.T)

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

        # project
        g_proj = U.T @ g

        # step
        dir_proj = self.inner_step_tensors("basis_opt", tensors=[g_proj], clone=False, grads=[g_proj])[0]

        # unproject
        update = U @ dir_proj

        # update = (U * L.pow(setting["matrix_power"])) @ z
        return update.view_as(tensor)

