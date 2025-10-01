from collections.abc import Callable
import torch
from .linalg_utils import mm
from . import torch_linalg
from .orthogonalize import orthogonalize, OrthogonalizeMethod
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


def regularize_eig(
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

def low_rank_eig_plus_sr1(
    L: torch.Tensor,
    Q: torch.Tensor,
    u: torch.Tensor,
    tol: float = 1e-9,
    retry_float64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    compute eigendecomposition of Q L Q^T + v v^T where L and Q are of a low rank matrix
    Q is ``(m, rank)`` and L is ``(rank, )``
    """
    z = Q.T @ u  # (rank,)

    # component of u orthogonal to the column space of Q
    h = u - Q @ z # (m,)
    beta = torch.linalg.vector_norm(h) # pylint:disable=not-callable

    if beta < tol:
        # u is already in the column space of Q
        B = L.diag_embed() + z.outer(z) # (rank, rank)
        L_prime, S = torch_linalg.eigh(B, retry_float64=retry_float64)
        Q_prime = Q @ S
        return L_prime, Q_prime

    else:
        # normalize the orthogonal component to get a new orthonormal vector
        v = h / beta # (m, )

        # basis for our new subspace is [Q, v]
        # B is corrected matrix in new basis:
        # B = D + w w^T where D = diag(L_diag, 0) and w = [z; beta]
        D_diag = torch.cat([L, torch.tensor([0.0], device=Q.device, dtype=Q.dtype)])
        w = torch.cat([z, beta.unsqueeze(0)]) # Shape: (rank+1,)
        B = D_diag.diag_embed() + w.outer(w)

        L_prime, S = torch_linalg.eigh(B, retry_float64=retry_float64)

        # reconstruct the eigenvectors of the corrected matrix
        basis = torch.cat([Q, v.unsqueeze(-1)], dim=1) # (m, rank+1)
        Q_prime = basis @ S # (m, rank+1)

        return L_prime, Q_prime
