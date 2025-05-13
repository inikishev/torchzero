import warnings
from collections.abc import Callable

import torch

def eigvals_func(A: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    L, Q = torch.linalg.eigh(A) # pylint:disable=not-callable
    L = fn(L)
    return  (Q * L.unsqueeze(-2)) @ Q.mH

def singular_vals_func(A: torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    U, S, V = torch.linalg.svd(A) # pylint:disable=not-callable
    S = fn(S)
    return (U * S.unsqueeze(-2)) @ V.mT

def matrix_power_eigh(A: torch.Tensor, pow:float):
    L, Q = torch.linalg.eigh(A) # pylint:disable=not-callable
    if pow % 2 != 0: L.clip_(min = torch.finfo(A.dtype).eps)
    return (Q * L.pow(pow).unsqueeze(-2)) @ Q.mH


def inv_sqrt_2x2(A: torch.Tensor, force_pd: bool=False) -> torch.Tensor:
    """Inverse square root of a possibly batched 2x2 matrix using a general formula for 2x2 matrices so that this is way faster than torch linalg. I tried doing a hierarchical 2x2 preconditioning but it didn't work well."""
    eps = torch.finfo(A.dtype).eps

    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    det = (a * d).sub_(b * c)
    trace = a + d

    if force_pd:
        # add smallest eigenvalue magnitude to diagonal to force PD
        # could also abs or clip eigenvalues bc there is a formula for eigenvectors
        term1 = trace/2
        term2 = (trace.pow(2).div_(4).sub_(det)).clamp_(min=eps).sqrt_()
        y1 = term1 + term2
        y2 = term1 - term2
        smallest_eigval = torch.minimum(y1, y2).neg_().clamp_(min=0) + eps
        a = a+smallest_eigval
        d = d+smallest_eigval

        # recalculate det and trace witg new a and b
        det = (a * d).sub_(b * c)
        trace = a + d

    s = (det.clamp(min=eps)).sqrt_()

    tau_squared = trace + 2 * s
    tau = (tau_squared.clamp(min=eps)).sqrt_()

    denom = s * tau

    coeff = (denom.clamp(min=eps)).reciprocal_().unsqueeze(-1).unsqueeze(-1)

    row1 = torch.stack([d + s, -b], dim=-1)
    row2 = torch.stack([-c, a + s], dim=-1)
    M = torch.stack([row1, row2], dim=-2)

    return coeff * M


def x_inv(diag: torch.Tensor,antidiag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """invert a matrix with diagonal and anti-diagonal non zero elements, with no checks that it is invertible"""
    n = diag.shape[0]
    if diag.dim() != 1 or antidiag.dim() != 1 or antidiag.shape[0] != n:
        raise ValueError("Input tensors must be 1D and have the same size.")
    if n == 0:
        return torch.empty_like(diag), torch.empty_like(antidiag)

    # opposite indexes
    diag_rev = torch.flip(diag, dims=[0])
    antidiag_rev = torch.flip(antidiag, dims=[0])

    # determinants
    # det_i = d[i] * d[n-1-i] - a[i] * a[n-1-i]
    determinant_vec = diag * diag_rev - antidiag * antidiag_rev

    # inverse diagonal elements: y_d[i] = d[n-1-i] / det_i
    inv_diag_vec = diag_rev / determinant_vec

    # inverse anti-diagonal elements: y_a[i] = -a[i] / det_i
    inv_anti_diag_vec = -antidiag / determinant_vec

    return inv_diag_vec, inv_anti_diag_vec