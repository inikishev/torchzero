from collections.abc import Callable

import torch

from . import torch_linalg
from .linalg_utils import mm
from .orthogonalize import OrthogonalizeMethod, orthogonalize
from .svd import tall_reduced_svd_via_eigh


# https://arxiv.org/pdf/2110.02820
def nystrom_approximation(
    A_mv: Callable[[torch.Tensor], torch.Tensor] | None,
    A_mm: Callable[[torch.Tensor], torch.Tensor] | None,
    ndim: int,
    rank: int,
    device,
    orthogonalize_method: OrthogonalizeMethod = 'qr',
    eigv_tol: float = 0,
    dtype = torch.float32,
    generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Nyström approximation to positive-semidefinite A factored as Q L Q^T (truncatd eigenvalue decomp),
    returns ``(L, Q)``.

    A is ``(m,m)``, then Q is ``(m, rank)``; L is a ``(rank, )`` vector - diagonal of ``(rank, rank)``"""
    # basis
    O = torch.randn((ndim, rank), device=device, dtype=dtype, generator=generator) # Gaussian test matrix
    O = orthogonalize(O, method=orthogonalize_method) # Thin QR decomposition # pylint:disable=not-callable

    # Y = AΩ
    AO = mm(A_mv=A_mv, A_mm=A_mm, X=O)

    v = torch.finfo(dtype).eps * torch.linalg.matrix_norm(AO, ord='fro') # Compute shift # pylint:disable=not-callable
    Yv = AO + v*O # Shift for stability
    C = torch.linalg.cholesky_ex(O.mT @ Yv)[0] # pylint:disable=not-callable
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

def eigh_plus_UUT(
    L: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    alpha: float = 1,
    tol = None,
    retry_float64: bool = False,
):
    """
    compute eigendecomposition of Q L Q^T + alpha * (U U^T), where Q is ``(m, rank)`` and L is ``(rank, )``,
    U is ``(m, k)`` where k is rank of correction
    """
    if U.size(1) == 1:
        return eigh_plus_uuT(L, Q, U[:,0], alpha=alpha, tol=tol, retry_float64=retry_float64)

    if tol is None: tol = torch.finfo(Q.dtype).eps
    m, r = Q.shape

    Z = Q.T @ U  # (r, k)
    U_res = U - Q @ Z  # (m, k)

    # find cols of U not in col space of Q
    res_norms = torch.linalg.vector_norm(U_res, dim=0) # pylint:disable=not-callable
    new_indices = torch.where(res_norms > tol)[0]
    k_prime = len(new_indices)

    if k_prime == 0:
        # all cols are in Q
        B = Q
        C = Z # (r x k)
        r_new = r
    else:
        # orthonormalize directions that aren't in Q
        U_new = U_res[:, new_indices]
        Q_u, _ = torch_linalg.qr(U_new, mode='reduced', retry_float64=retry_float64)
        B = torch.hstack([Q, Q_u])
        C = torch.vstack([Z, Q_u.T @ U])
        r_new = r + k_prime


    # project and compute new eigendecomposition
    A_proj = torch.zeros((r_new, r_new), device=Q.device, dtype=Q.dtype)
    A_proj[:r, :r] = L.diag_embed()
    A_proj.addmm_(C, C.T, alpha=alpha)

    L_prime, S = torch_linalg.eigh(A_proj, retry_float64=retry_float64)

    # unproject and sort
    Q_prime = B @ S
    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime


def eigh_plus_UVT_symmetrize(
    Q: torch.Tensor,
    L: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    alpha: float,
    retry_float64: bool = False,

):
    """
    Q is ``(m, rank)``; L is ``(rank, )``; U and V are the low rank correction such that U V^T is ``(m, m)``.

    This computes eigendecomposition of A, where

    ``M = Q diag(L) Q^T + alpha * (U V^T)``;

    ``A = (M + M^T) / 2``
    """
    m, rank = Q.shape
    _, k = V.shape

    # project U and V out of the Q subspace via Gram-schmidt
    Q_T_U = Q.T @ U
    U_perp = U - Q @ Q_T_U

    Q_T_V = Q.T @ V
    V_perp = V - Q @ Q_T_V

    R = torch.hstack([U_perp, V_perp])
    Q_perp, _ = torch_linalg.qr(R, retry_float64=retry_float64)

    Q_B = torch.hstack([Q, Q_perp])
    r_B = Q_B.shape[1]

    # project, symmetrize and compute new eigendecomposition
    A_proj = torch.zeros((r_B, r_B), device=Q.device, dtype=Q.dtype)
    A_proj[:rank, :rank] = L.diag_embed()

    Q_perp_T_U = Q_perp.T @ U
    Q_B_T_U = torch.vstack([Q_T_U, Q_perp_T_U])

    Q_perp_T_V = Q_perp.T @ V
    Q_B_T_V = torch.vstack([Q_T_V, Q_perp_T_V])

    update_proj = Q_B_T_U @ Q_B_T_V.T + Q_B_T_V @ Q_B_T_U.T
    A_proj.add_(update_proj, alpha=alpha/2)

    L_prime, S = torch_linalg.eigh(A_proj, retry_float64=retry_float64)

    # unproject and sort
    Q_prime = Q_B @ S

    idx = torch.argsort(L_prime)
    L_prime = L_prime[idx]
    Q_prime = Q_prime[:, idx]

    return L_prime, Q_prime
