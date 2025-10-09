from collections.abc import Callable

import torch

from . import torch_linalg
from .linalg_utils import mm
from .orthogonalize import OrthogonalizeMethod, orthogonalize
from .svd import tall_reduced_svd_via_eigh


# https://arxiv.org/pdf/2110.02820
def nystrom_approximation(
    Omega: torch.Tensor,
    AOmega: torch.Tensor,
    eigv_tol: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Nyström approximation to positive-semidefinite A factored as Q L Q^T (truncatd eigenvalue decomp),
    returns ``(L, Q)``.

    A is ``(m,m)``, then Q is ``(m, rank)``; L is a ``(rank, )`` vector - diagonal of ``(rank, rank)``"""

    v = torch.finfo(AOmega.dtype).eps * torch.linalg.matrix_norm(AOmega, ord='fro') # Compute shift # pylint:disable=not-callable
    Yv = AOmega + v*Omega # Shift for stability
    C = torch.linalg.cholesky_ex(Omega.mT @ Yv)[0] # pylint:disable=not-callable
    B = torch.linalg.solve_triangular(C, Yv.mT, upper=False, unitriangular=False).mT # pylint:disable=not-callable

    # Q, S, _ = torch_linalg.svd(B, full_matrices=False) # pylint:disable=not-callable
    # B is (ndim, rank) so we can use eigendecomp of (rank, rank)
    Q, S = tall_reduced_svd_via_eigh(B, tol=eigv_tol, retry_float64=True)

    L = S.pow(2) - v
    return L, Q


def regularize_eigh(
    L: torch.Tensor,
    Q: torch.Tensor,
    truncate: int | None = None,
    tol: float | None = None,
    damping: float = 0,
    rdamping: float = 0,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Applies regularization to eigendecomposition. Returns ``(L, Q)``.

    Args:
        L (torch.Tensor): eigenvalues, shape ``(rank,)``.
        Q (torch.Tensor): eigenvectors, shape ``(n, rank)``.
        truncate (int | None, optional):
            keeps top ``truncate`` eigenvalues. Defaults to None.
        tol (float | None, optional):
            all eigenvalues smaller than largest eigenvalue times ``tol`` are removed. Defaults to None.
        damping (float | None, optional): scalar added to eigenvalues. Defaults to 0.
        rdamping (float | None, optional): scalar multiplied by largest eigenvalue and added to eigenvalues. Defaults to 0.
    """
    # remove non-finite eigenvalues
    finite = L.isfinite()
    if finite.any():
        L = L[finite]
        Q = Q[:, finite]
    else:
        return None, None

    # largest finite!!! eigval
    L_max = L[-1] # L is sorted in ascending order

    # remove small eigenvalues relative to largest
    if tol is not None:
        indices = L > tol * L_max
        L = L[indices]
        Q = Q[:, indices]

    # truncate to rank (L is ordered in ascending order)
    if truncate is not None:
        L = L[-truncate:]
        Q = Q[:, -truncate:]

    # damping
    d = damping + rdamping * L_max
    if d != 0:
        L += d

    return L, Q

def eigh_plus_uuT(
    L: torch.Tensor,
    Q: torch.Tensor,
    u: torch.Tensor,
    alpha: float = 1,
    tol: float | None = None,
    retry_float64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    compute eigendecomposition of Q L Q^T + alpha * (u u^T) where Q is ``(m, rank)`` and L is ``(rank, )`` and u is ``(m, )``
    """
    if tol is None: tol = torch.finfo(Q.dtype).eps
    z = Q.T @ u  # (rank,)

    # component of u orthogonal to the column space of Q
    res = u - Q @ z # (m,)
    beta = torch.linalg.vector_norm(res) # pylint:disable=not-callable

    if beta < tol:
        # u is already in the column space of Q
        B = L.diag_embed().add_(z.outer(z), alpha=alpha) # (rank, rank)
        L_prime, S = torch_linalg.eigh(B, retry_float64=retry_float64)
        Q_prime = Q @ S
        return L_prime, Q_prime

    # normalize the orthogonal component to get a new orthonormal vector
    v = res / beta # (m, )

    # project and compute new eigendecomposition
    D_diag = torch.cat([L, torch.tensor([0.0], device=Q.device, dtype=Q.dtype)])
    w = torch.cat([z, beta.unsqueeze(0)]) # Shape: (rank+1,)
    B = D_diag.diag_embed().add_(w.outer(w), alpha=alpha)

    L_prime, S = torch_linalg.eigh(B, retry_float64=retry_float64)

    # unproject and sort
    basis = torch.cat([Q, v.unsqueeze(-1)], dim=1) # (m, rank+1)
    Q_prime = basis @ S # (m, rank+1)

    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime

def eigh_plus_UUt(
    L: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    alpha: float | torch.Tensor = 1,
    tol = None,
    ortho_method: OrthogonalizeMethod = 'qr',
    retry_float64=True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    compute eigendecomposition of Q L Q^T + alpha * (U U^T), where Q is ``(m, rank)`` and L is ``(rank, )``,
    U is ``(m, k)`` where k is rank of correction

    returns ``(L, Q)``
    """
    if U.size(1) == 1:
        return eigh_plus_uuT(L, Q, U[:,0], alpha=float(alpha), tol=tol)

    # make alpha shape (k, )
    k = U.size(1)
    if isinstance(alpha, torch.Tensor):
        alpha = torch.broadcast_to(alpha, (k, ))
    else:
        alpha = torch.full((k,), float(alpha), device=U.device, dtype=U.dtype)

    if tol is None: tol = torch.finfo(Q.dtype).eps
    m, r = Q.shape
    QtU = Q.T @ U  # (r, k)
    U_res = U - Q @ QtU  # (m, k)

    # find cols of U not in col space of Q
    res_norms = torch.linalg.vector_norm(U_res, dim=0) # pylint:disable=not-callable
    new_indices = torch.where(res_norms > tol)[0]
    k_prime = len(new_indices)

    if k_prime == 0:
        # all cols are in Q
        B = Q
        C = QtU # (r x k)
        r_new = r
    else:
        # orthonormalize directions that aren't in Q
        U_new = U_res[:, new_indices]
        Q_u = orthogonalize(U_new, method=ortho_method)
        B = torch.hstack([Q, Q_u])
        C = torch.vstack([QtU, Q_u.T @ U_res])
        r_new = r + k_prime

    # project and compute new eigendecomposition
    A_proj = torch.zeros((r_new, r_new), device=Q.device, dtype=Q.dtype)
    A_proj[:r, :r] = L.diag_embed()
    # A_proj += (C @ C.T).mul_(alpha)
    A_proj.addmm_(C * alpha, C.T)

    try:
        L_prime, S = torch_linalg.eigh(A_proj, retry_float64=retry_float64)
    except torch.linalg.LinAlgError:
        return None, None

    # unproject and sort
    Q_prime = B @ S
    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime


def eigh_plus_UUt_mm(
    # A1 = Q @ diag(L) @ Q.T
    L: torch.Tensor,
    Q: torch.Tensor,

    # A2 = U @ U.T
    U: torch.Tensor,

    # rhs
    B: torch.Tensor,

    # weights
    w1: float,
    w2: float | torch.Tensor,

) -> torch.Tensor:
    """
    Computes ``(w1 * (Q L Q^T) + (U diag(w2) U^T) @ B``,

    Q is ``(m, rank)``, L is ``(rank, rank)``, U is ``(m, z)``, B is ``(m, k)``.

    Returns ``(m, k)``
    """
    # sketch Q L Q^T
    QtB = Q.T @ B # (rank, k)
    LQtB = L.unsqueeze(1) * QtB  # (rank, k)
    sketch1 = Q @ LQtB  # (m, k)

    # skecth U U^T
    UtB = U.T @ B # (z, k)
    if isinstance(w2, torch.Tensor) and w2.numel() > 1: w2UtB = w2.unsqueeze(-1) * UtB
    else:  w2UtB = w2 * UtB
    sketch2 = U @ w2UtB # (m, k)

    return w1 * sketch1 + sketch2


def randomized_eigh_plus_UUt(
    L1: torch.Tensor,
    Q1: torch.Tensor,
    U: torch.Tensor,
    w1: float,
    w2: float | torch.Tensor,
    oversampling_p: int,
    rank: int,
    eig_tol: float,
    damping: float,
    rdamping: float,
    ortho_method: OrthogonalizeMethod = 'qr',
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    compute randomized eigendecomposition of w1 * Q L Q^T + w2 * (U U^T),
    where Q is ``(m, rank)`` and L is ``(rank, )``,
    U is ``(m, k)`` where k is rank of correction, returns ``(L, Q)``
    """
    n = Q1.shape[0]
    device = Q1.device
    dtype = Q1.dtype
    l = rank + oversampling_p

    # gaussian test matrix
    Omega = torch.randn(n, l, device=device, dtype=dtype)

    # sketch
    AOmega = eigh_plus_UUt_mm(L1, Q1, U, Omega, w1, w2)
    Q = orthogonalize(AOmega, ortho_method)

    AQ = eigh_plus_UUt_mm(L1, Q1, U, Q, w1, w2)
    QtAQ = Q.T @ AQ

    W = (QtAQ + QtAQ.T) / 2.0

    # compute new L and Q
    try:
        L_prime, S = torch.linalg.eigh(W) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        return L1, Q1

    L_prime, S = regularize_eigh(L=L_prime, Q=S, truncate=rank, tol=eig_tol, damping=damping, rdamping=rdamping)

    if L_prime is None or S is None:
        return L1, Q1

    return L_prime, Q @ S


def rank1_eigh(v: torch.Tensor):
    """returns ``(L, Q)`` of ``(v v^T)``"""
    vv = v.dot(v)
    norm = vv.sqrt().clip(min=torch.finfo(vv.dtype).tiny * 2)

    L = vv.unsqueeze(0) # (rank, )
    Q = v.unsqueeze(-1) / norm # (m, rank)

    return L, Q

def low_rank_eigh(U: torch.Tensor):
    """returns ``(L, Q)`` of ``alpha * (U U^T)`` (from GGT)"""
    M = U.T @ U
    L, S = torch.linalg.eigh(M) # pylint:disable=not-callable

    Q = U @ S
    Q /= torch.sqrt(L).clip(min=torch.finfo(L.dtype).tiny * 2)

    return L, Q
