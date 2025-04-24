import warnings

import torch


# sqrtmh function from https://github.com/pytorch/pytorch/issues/25481#issuecomment-1109537907
def sqrtmh(A):
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices"""
    L, Q = torch.linalg.eigh(A) # pylint:disable=not-callable
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return ((Q * L.sqrt().unsqueeze(-2)) @ Q.mH).real



def matrix_inverse_root(matrix: torch.Tensor, exponent: float, eps: float = 1e-6) -> torch.Tensor:
    """Computes matrix^(-1/exponent) using eigenvalue decomposition"""
    if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"Input must be a square matrix or batch of square matrices. Got shape {matrix.shape}")

    try:
        d = matrix.shape[-1]
        matrix = matrix + torch.eye(d, device=matrix.device, dtype=matrix.dtype) * eps

        eigvals, eigvecs = torch.linalg.eigh(matrix) # pylint:disable=not-callable
        eigvals = torch.clamp(eigvals, min=eps)
        inv_root_eigvals = eigvals.pow_(-1.0 / exponent)

        # reconstruct
        inv_root_matrix = eigvecs @ torch.diag_embed(inv_root_eigvals) @ eigvecs.mH

        # needed?
        if not matrix.is_complex() and inv_root_matrix.is_complex():
            inv_root_matrix = inv_root_matrix.real

        return inv_root_matrix

    except torch.linalg.LinAlgError:
        # Fallback if eigh fails (e.g., matrix not PSD even with epsilon)
        # Using pseudo-inverse might be another option.
        warnings.warn("linalg.eigh failed in _compute_inv_root. Returning identity.", RuntimeWarning)
        return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)