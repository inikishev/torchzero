# pylint: disable = non-ascii-name
import torch

from ...core import Chainable, TensorTransform
from ...linalg import (
    OrthogonalizeMethod,
    orthogonalize,
    regularize_eigh,
    torch_linalg,
)
from ...linalg.linear_operator import Eigendecomposition
from ..adaptive.lre_optimizers import LREOptimizerBase
from .eigengrad import _eigengrad_update_state_, eigengrad_apply


def weighted_eigen_plus_rank1_mm(
    # A1 = Q1 @ diag(L1) @ Q1.T
    L1: torch.Tensor,
    Q1: torch.Tensor,

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
        L1 (torch.Tensor): eigenvalues of A1, shape ``(rank,)``.
        Q1 (torch.Tensor): eigenvectors of A1, shape ``(n, rank)``.
        v2 (torch.Tensor): vector such that ``v v^T = A2``, shape ``(n,)``.
        B (torch.Tensor): shape ``(n, k)``.
        w1 (float): weight for A1.
        w2 (float): weight for A2.

    """
    # sketch A1
    QTB = Q1.T @ B # (rank, k)
    LQTB = L1.unsqueeze(1) * QTB  # (rank, k)
    sketch1 = Q1 @ LQTB  # (n, k)

    # skecth A2
    vB = v2 @ B
    sketch2 = v2.outer(vB)

    return w1 * sketch1 + w2 * sketch2


def adanystrom_update(
    L1: torch.Tensor,
    Q1: torch.Tensor,
    v2: torch.Tensor,
    w1: float,
    w2: float,
    oversampling_p: int,
    rank: int,
    eig_tol: float,
    damping: float,
    rdamping: float,
    orthogonalize_method: OrthogonalizeMethod,

) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """computes the Nyström approximation of ``(w1 * A1 + w2 * A2)``,
    where ``A1`` is an eigendecomposition, ``A2`` is symmetric rank 1.

    returns L of shape ``(k, )`` and Q of shape ``(n, k)``.

    Args:
        L1 (torch.Tensor): eigenvalues of A1, shape ``(rank,)``.
        Q1 (torch.Tensor): eigenvectors of A1, shape ``(n, rank)``.
        v2 (torch.Tensor): vector such that ``v v^T = A2``, shape ``(n,)`` or ``(n, 1)``.
        w1 (float): weight for A1.
        w2 (float): weight for A2.
    """
    n = Q1.shape[0]
    device = Q1.device
    dtype = Q1.dtype
    l = rank + oversampling_p

    # gaussian test matrix
    Omega = torch.randn(n, l, device=device, dtype=dtype)

    # sketch
    AOmega = weighted_eigen_plus_rank1_mm(L1, Q1, v2, Omega, w1, w2)
    Q = orthogonalize(AOmega, orthogonalize_method)

    AQ = weighted_eigen_plus_rank1_mm(L1, Q1, v2, Q, w1, w2)
    QTAQ = Q.T @ AQ

    W = (QTAQ + QTAQ.T) / 2.0

    # compute new L and Q
    try:
        L_prime, S = torch_linalg.eigh(W, retry_float64=True)
    except torch.linalg.LinAlgError:
        return L1, Q1

    L_prime, S = regularize_eigh(L=L_prime, Q=S, truncate=rank, tol=eig_tol, damping=damping, rdamping=rdamping)

    if L_prime is None or S is None:
        return L1, Q1

    return L_prime, Q @ S


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
        eig_tol (float, optional):
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
        concat_params (bool, optional):
            whether to precondition all parameters at once if True, or each separately if False. Defaults to True.
        update_freq (int, optional): update frequency. Defaults to 1.
        inner (Chainable | None, optional): inner modules. Defaults to None.
    """
    def __init__(
        self,
        rank:int = 100,
        beta=0.95,
        oversampling: int = 10,
        eig_tol: float | None = 1e-32,
        damping: float = 0,
        rdamping: float = 0,
        mm_tol: float = 0,
        mm_truncate: int | None = None,
        mm_damping: float = 0,
        mm_rdamping: float = 0,
        id_reg: float | None = None,
        orthogonalize_method: OrthogonalizeMethod = 'qr',
        eigenbasis_optimizer: LREOptimizerBase | None = None,
        orthogonalize_interval: int | None = 100,

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
        rank = setting["rank"]
        device = tensor.device
        dtype = tensor.dtype
        beta = setting["beta"]

        try:
            if "L" not in state:
                # use just tensor and zero L and Q with zero weight

                L, Q = adanystrom_update(
                    L1=torch.zeros(rank, device=device, dtype=dtype),
                    Q1=torch.zeros((tensor.numel(), rank), device=device, dtype=dtype),
                    v2=tensor.ravel(),
                    w1=0,
                    w2=1-beta,
                    rank=rank,
                    oversampling_p=setting["oversampling"],
                    eig_tol=setting["eig_tol"],
                    damping=setting["damping"],
                    rdamping=setting["rdamping"],
                    orthogonalize_method=setting["orthogonalize_method"],
                )

                state["L"] = state["L_reg"] = L
                state["Q"] = state["Q_reg"] = Q

            else:
                L = state["L"]
                Q = state["Q"]

                w1 = beta
                w2 = 1 - w1

                # compute new factors (this function truncates them)
                L_new, Q_new = adanystrom_update(
                    L1=L,
                    Q1=Q,
                    v2=tensor.ravel(),
                    w1=w1,
                    w2=w2,
                    rank=rank,
                    oversampling_p=setting["oversampling"],
                    eig_tol=setting["eig_tol"],
                    damping=setting["damping"],
                    rdamping=setting["rdamping"],
                    orthogonalize_method=setting["orthogonalize_method"],
                )

                _eigengrad_update_state_(state=state, setting=setting, L_new=L_new, Q_new=Q_new)

        except torch.linalg.LinAlgError:
            pass

    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):
        if "L_reg" not in state:
            return tensor.clip(-0.1, 0.1)

        if "eigenbasis_state" not in state:
            state["eigenbasis_state"] = {}

        return eigengrad_apply(
            tensor=tensor,
            L_reg = state["L_reg"],
            Q_reg = state["Q_reg"],
            beta = setting["beta"],
            step = state["step"],
            debias = True,
            id_reg = setting["id_reg"],
            eigenbasis_optimizer = setting["eigenbasis_optimizer"],
            eigenbasis_state = state["eigenbasis_state"]
        )
