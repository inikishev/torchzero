# pylint: disable = non-ascii-name
import torch

from ...core import Chainable, TensorTransform
from ...linalg.eigh import low_rank_eig_plus_sr1, regularize_eig
from ...linalg.linear_operator import Eigendecomposition
from ..adaptive.subspace_optimizers import SubspaceOptimizerBase

class Eigengrad(TensorTransform):
    """we can easily compute rank 1 symmetric update to a low rank eigendecomposition.
    So this stores covariance matrix as it.


    Args:
        rank (int): maximum allowed rank
        beta (float, optional): beta for covariance matrix exponential moving average. Defaults to 0.95.
        tol (float, optional):
            removes eigenvalues this much smaller than largest eigenvalue when updating the preconditioner. Defaults to 1e-7.
        damping (float, optional):
            added to eigenvalues when updating the preconditioner. Defaults to 1e-8.
        rdamping (float, optional):
            added to eigenvalues when updating the preconditioner, relative to largest eigenvalue. Defaults to 0.
        mm_tol (float, optional):
            removes eigenvalues this much smaller than largest eigenvalue when computing the update. Defaults to 1e-7.
        mm_truncate (int | None, optional):
            uses top k eigenvalues to compute the update. Defaults to None.
        mm_damping (float, optional):
            added to eigenvalues when computing the update. Defaults to 1e-4.
        mm_rdamping (float, optional):
            added to eigenvalues when computing the update, relative to largest eigenvalue. Defaults to 0.
        id_reg (float, optional):
            multiplier to identity matrix added to preconditioner before computing update
            If this value is given, solution from Nyström sketch-and-solve will be used to compute the update.
            This value can't be too small (i.e. less than 1e-5) or the solver will be very unstable. Defaults to None.
        column_space_tol (float, optional):
            tolerance for deciding if new eigenvector is within column space of the covariance matrix. Defaults to 1e-9.
        concat_params (bool, optional):
            whether to precondition all parameters at once if True, or each separately if False. Defaults to True.
        update_freq (int, optional): update frequency. Defaults to 1.
        inner (Chainable | None, optional): inner modules. Defaults to None.

    """

    def __init__(
        self,
        rank: int,
        beta=0.95,
        tol: float = 1e-7,
        damping: float = 1e-8,
        rdamping: float = 0,
        mm_tol: float = 0,
        mm_truncate: int | None = None,
        mm_damping: float = 1e-4,
        mm_rdamping: float = 0,
        id_reg: float | None = None,
        column_space_tol=1e-9,
        subspace_optimizer: SubspaceOptimizerBase | None = None,
        concat_params: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        for k in ["self", "concat_params", "inner", "update_freq"]:
            del defaults[k]

        super().__init__(defaults, concat_params=concat_params, inner=inner, update_freq=update_freq)

    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        state["step"] = state.get("step", 0) + 1

        try:
            if "L" not in state:
                # for uu^T u is eigenvector and u^T u is eigenvalue
                norm = torch.linalg.vector_norm(tensor).clip(min=torch.finfo(tensor.dtype).tiny * 2) # pylint:disable=not-callable

                state["L"] = tensor.dot(tensor).unsqueeze(0) / norm # (rank,)
                state["Q"] = tensor.unsqueeze(-1) / norm # (m, rank)

            else:
                beta = setting["beta"]
                L = state["L"]
                Q = state["Q"]

                # compute new factors
                L_new, Q_new = low_rank_eig_plus_sr1(L*beta, Q, tensor*(1-beta), tol=setting["column_space_tol"], retry_float64=True)

                # truncate/regularize new factors
                L_new, Q_new = regularize_eig(L_new, Q_new, truncate=setting["rank"], tol=setting["tol"],
                                              damping=setting["damping"], rdamping=setting["rdamping"])

                # reproject subspace optimizer
                subspace_optimizer: SubspaceOptimizerBase | None = setting["subspace_optimizer"]
                if subspace_optimizer is not None:
                    if (L_new is not None) and (Q_new is not None):
                        subspace_state = state["subspace_state"]
                        subspace_optimizer.reproject(L_old=L, Q_old=Q, L_new=L_new, Q_new=Q_new, state=subspace_state)

                # store
                if L_new is not None: state["L"] = L_new
                if Q_new is not None: state["Q"] = Q_new

        except torch.linalg.LinAlgError:
            pass

    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        if "L" not in state:
            return tensor.clip(-0.1, 0.1)

        L = state["L"]
        Q = state["Q"]
        id_reg = setting["id_reg"]

        # debias
        # we don't start from zeros though so maybe this isn't necessary but its just warmup
        beta = setting["beta"]
        L = L / (1 - beta**state["step"])

        # regularize for matmul
        L, Q = regularize_eig(
            L=L,
            Q=Q,
            truncate=setting["mm_truncate"],
            tol=setting["mm_tol"],
            damping=setting["mm_damping"],
            rdamping=setting["mm_rdamping"],
        )

        if L is None or Q is None:
            del state["L"], state["Q"]
            return tensor.clip(-0.1, 0.1)

        # step with subspace optimizer
        subspace_optimizer: SubspaceOptimizerBase | None = setting["subspace_optimizer"]
        if subspace_optimizer is not None:
            if (id_reg is not None) and (id_reg != 0):
                raise RuntimeError("id_reg is not compatible with subspace_optimizer")

            if "subspace_state" not in state: state["subspace_state"] = {}
            subspace_state = state["subspace_state"]

            update = subspace_optimizer.step(tensor.ravel(), L=L, Q=Q, state=subspace_state)
            return update.view_as(tensor)

        # or just whiten
        L = L.clip(min=torch.finfo(L.dtype).tiny * 2)

        if id_reg is None or id_reg == 0:
            G = Eigendecomposition(L.sqrt(), Q, use_nystrom=False)
            dir = G.solve(tensor.ravel())

        else:
            G = Eigendecomposition(L.sqrt(), Q, use_nystrom=True)
            dir = G.solve_plus_diag(tensor.ravel(), diag=id_reg)

        return dir.view_as(tensor)
