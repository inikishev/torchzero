# pylint: disable = non-ascii-name
import torch

from ...core import Chainable, TensorTransform
from ...linalg import (
    OrthogonalizeMethod,
    nystrom_approximation,
    orthogonalize, regularize_eig,
    torch_linalg,
)
from ...linalg.linear_operator import Eigendecomposition


def weighted_eigen_plus_rank1_mm(
    # A1 = P1 @ diag(D1) @ P1.T
    D1: torch.Tensor,
    P1: torch.Tensor,

    # K2 = v2 @ v2.T
    v2: torch.Tensor,

    # second matrix
    B: torch.Tensor,

    # weights
    w1: float,
    w2: float,

) -> torch.Tensor:
    """
    Computes ``(w1 * A1 + w2 * A2) @ B``, where ``A1`` is an eigendecomposition, ``A2`` is symmetric rank 1.

    Returns ``(n, k)``

    Args:
        D1 (torch.Tensor): eigenvalues of A1, shape ``(rank,)``.
        P1 (torch.Tensor): eigenvectors of A1, shape ``(n, rank)``.
        v2 (torch.Tensor): vector such that ``v v^T = A2``, shape ``(n,)`` or ``(n, 1)``.
        B (torch.Tensor): shape ``(n, k)``.
        w1 (float): weight for A1.
        w2 (float): weight for A2.

    """
    if v2.ndim == 1:
        v2 = v2.unsqueeze(1)

    # sketch A1
    PᵀB = D1.T @ B # (rank, k)
    DPᵀB = P1.unsqueeze(1) * PᵀB  # (rank, k)
    sketch1 = D1 @ DPᵀB # (n, k)

    # skecth A2
    vB = v2.T @ B
    sketch2 = v2 @ vB

    return w1 * sketch1 + w2 * sketch2


def adanystrom_update(
    D1: torch.Tensor,
    P1: torch.Tensor,
    v2: torch.Tensor,
    w1: float,
    w2: float,
    oversampling_p: int,
    rank: int,
    tol: float,
    damping: float,
    rdamping: float,
    orthogonalize_method: OrthogonalizeMethod,

) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """computes the Nyström approximation of ``(w1 * A1 + w2 * A2)``,
    where ``A1`` is an eigendecomposition, ``A2`` is symmetric rank 1.

    returns D of shape ``(k, )`` and P of shape ``(n, k)``.

    Args:
        D1 (torch.Tensor): eigenvalues of A1, shape ``(rank,)``.
        P1 (torch.Tensor): eigenvectors of A1, shape ``(n, rank)``.
        v2 (torch.Tensor): vector such that ``v v^T = A2``, shape ``(n,)`` or ``(n, 1)``.
        w1 (float): weight for A1.
        w2 (float): weight for A2.
    """
    n = D1.shape[0]
    device = D1.device
    dtype = D1.dtype
    l = rank + oversampling_p

    # gaussian test matrix
    Ω = torch.randn(n, l, device=device, dtype=dtype)

    # sketch
    AΩ = weighted_eigen_plus_rank1_mm(D1, P1, v2, Ω, w1, w2)
    Q = orthogonalize(AΩ, orthogonalize_method)

    AQ = weighted_eigen_plus_rank1_mm(D1, P1, v2, Q, w1, w2)
    AᵀAQ = Q.T @ AQ

    W = (AᵀAQ + AᵀAQ.T) / 2.0

    # compute new L and Q
    try:
        D, P = torch_linalg.eigh(W, retry_float64=True)
    except torch.linalg.LinAlgError:
        return D1, P1

    D, P = regularize_eig(L=D, Q=P, truncate=rank, tol=tol, damping=damping, rdamping=rdamping)

    if D is None or P is None:
        return D1, P1

    return D, Q @ P


# def adanystrom_update2(
#     L1: torch.Tensor,
#     Q1: torch.Tensor,
#     v2: torch.Tensor,
#     w1: float,
#     w2: float,
#     rank: int,
# ):
#     def A_mm(X):
#         return weighted_eigen_plus_rank1_mm(L1=L1, Q1=Q1, v2=v2, B=X, w1=w1, w2=w2)

#     return nystrom_approximation(A_mm, A_mm=A_mm, ndim=v2.numel(), rank=rank, device=L1.device, dtype=L1.dtype)

class AdaNystrom(TensorTransform):
    """Adagrad/RMSprop/Adam with Nyström-approximated covariance matrix.

    Args:
        rank (_type_): rank of Nyström approximation.
        w1 (float, optional): weight of current covariance matrix. Defaults to 0.95.
        w2 (float, optional): weight of new gradient in covariance matrix. Defaults to 0.05.
        oversampling (int, optional): number of extra random vectors (top rank eigenvalues are kept). Defaults to 10.
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
        concat_params (bool, optional):
            whether to precondition all parameters at once if True, or each separately if False. Defaults to True.
        update_freq (int, optional): update frequency. Defaults to 1.
        inner (Chainable | None, optional): inner modules. Defaults to None.
    """
    def __init__(
        self,
        rank,
        w1=0.95,
        w2=0.05,
        oversampling: int = 10,
        tol: float = 1e-7,
        damping: float = 1e-8,
        rdamping: float = 0,
        mm_tol: float = 0,
        mm_truncate: int | None = None,
        mm_damping: float = 1e-4,
        mm_rdamping: float = 0,
        nystrom_reg: float | None = None,
        orthogonalize_method: OrthogonalizeMethod = 'qr',
        concat_params: bool = True,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        for k in ["self", "concat_params", "inner", "update_freq"]:
            del defaults[k]

        super().__init__(defaults, concat_params=concat_params, inner=inner, update_freq=update_freq)

    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        rank = setting["rank"]
        device = tensor.device
        dtype = tensor.dtype

        try:
            if "D" not in state:
                # use just tensor and zero D and P

                state["D"], state["P"] = adanystrom_update(
                    D1=torch.zeros(tensor.numel(), rank, device=device, dtype=dtype),
                    P1=torch.zeros(rank, device=device, dtype=dtype),
                    v2=tensor.ravel(),
                    w1=0,
                    w2=1,
                    rank=rank,
                    oversampling_p=setting["oversampling"],
                    tol=setting["tol"],
                    damping=setting["damping"],
                    rdamping=setting["rdamping"],
                    orthogonalize_method=setting["orthogonalize_method"],
                )

            else:
                w1, w2 = setting["w1"],setting["w2"]
                state["D"], state["P"] = adanystrom_update(
                    D1=state["P"],
                    P1=state["D"],
                    v2=tensor.ravel(),
                    w1=w1,
                    w2=w2,
                    rank=rank,
                    oversampling_p=setting["oversampling"],
                    tol=setting["tol"],
                    damping=setting["damping"],
                    rdamping=setting["rdamping"],
                    orthogonalize_method=setting["orthogonalize_method"],
                )
        except torch.linalg.LinAlgError:
            pass

    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        if "D" not in state:
            return tensor.clip(-0.1, 0.1)

        D = state["D"]
        P = state["P"]

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
