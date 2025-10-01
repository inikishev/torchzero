# pylint: disable = non-ascii-name
import torch

from ...core import Chainable, TensorTransform
from ...linalg.eigh import low_rank_eig_plus_sr1, regularize_eig
from ...linalg.linear_operator import Eigendecomposition

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
        nystrom_reg (float, optional):
            multiplier to identity matrix added to preconditioner before computing update
            If this value is given, Nyström sketch-and-solve will be used to compute the update.
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
        nystrom_reg: float | None = None,
        column_space_tol=1e-9,
        concat_params: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        for k in ["self", "concat_params", "inner", "update_freq"]:
            del defaults[k]

        super().__init__(defaults, concat_params=concat_params, inner=inner, update_freq=update_freq)

    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        try:
            state["step"] = state.get("step", 0) + 1
            if "D" not in state:
                # for uu^T u is eigenvector and u^T u is eigenvalue
                norm = torch.linalg.vector_norm(tensor).clip(min=torch.finfo(tensor.dtype).tiny * 2) # pylint:disable=not-callable

                state["D"] = tensor.dot(tensor).unsqueeze(0) / norm # (rank,)
                state["P"] = tensor.unsqueeze(-1) / norm # (m, rank)

            else:
                beta = setting["beta"]
                D = state["D"]
                P = state["P"]

                # decay current eigenvalues
                D, P = low_rank_eig_plus_sr1(D*beta, P, tensor*(1-beta), tol=setting["column_space_tol"], retry_float64=True)

                # regularize accumulator
                D, P = regularize_eig(D, P, truncate=setting["rank"], tol=setting["tol"], damping=setting["damping"], rdamping=setting["rdamping"])

                # store
                state["D"] = D
                state["P"] = P

        except torch.linalg.LinAlgError:
            pass

    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        if "D" not in state:
            return tensor.clip(-0.1, 0.1)

        D = state["D"]
        P = state["P"]

        # debias
        # we don't start from zeros though so maybe this isn't necessary but its just warmup
        beta = setting["beta"]
        D = D / (1 - beta**state["step"])

        # regularize for matmul
        D, P = regularize_eig(
            L=D,
            Q=P,
            truncate=setting["mm_truncate"],
            tol=setting["mm_tol"],
            damping=setting["mm_damping"],
            rdamping=setting["mm_rdamping"],
        )

        if D is None or P is None:
            del state["D"], state["P"]
            return tensor.clip(-0.1, 0.1)

        D = D.clip(min=torch.finfo(D.dtype).tiny * 2)

        nystrom_reg = setting["nystrom_reg"]
        if nystrom_reg is None:
            G = Eigendecomposition(D.sqrt(), P, use_nystrom=False)
            dir = G.solve(tensor.ravel())

        else:
            G = Eigendecomposition(D.sqrt(), P, use_nystrom=True)
            dir = G.solve_plus_diag(tensor.ravel(), diag=nystrom_reg)

        return dir.view_as(tensor)
