from collections.abc import Callable
import torch
from .linalg_utils import mm



# https://arxiv.org/pdf/2110.02820
def nystrom_approximation(
    A_mv: Callable[[torch.Tensor], torch.Tensor] | None,
    A_mm: Callable[[torch.Tensor], torch.Tensor] | None,
    ndim: int,
    rank: int,
    device,
    dtype = torch.float32,
    generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Nyström approximation to positive-semidefinite A factored as Q L Q^T (truncatd eigenvalue decomp),
    returns ``(L, Q)``.

    A is ``(m,m)``, then Q is ``(m, rank)``; L is a ``(rank, )`` vector - diagonal of ``(rank, rank)``"""
    # basis
    O = torch.randn((ndim, rank), device=device, dtype=dtype, generator=generator) # Gaussian test matrix
    O, _ = torch.linalg.qr(O) # Thin QR decomposition # pylint:disable=not-callable

    # Y = AΩ
    AO = mm(A_mv=A_mv, A_mm=A_mm, X=O)

    v = torch.finfo(dtype).eps * torch.linalg.matrix_norm(AO, ord='fro') # Compute shift # pylint:disable=not-callable
    Yv = AO + v*O # Shift for stability
    C = torch.linalg.cholesky_ex(O.mT @ Yv)[0] # pylint:disable=not-callable
    B = torch.linalg.solve_triangular(C, Yv.mT, upper=False, unitriangular=False).mT # pylint:disable=not-callable
    Q, S, _ = torch.linalg.svd(B, full_matrices=False) # pylint:disable=not-callable
    L = (S.pow(2) - v).clip(min=0) #Remove shift, compute eigs
    return L, Q
