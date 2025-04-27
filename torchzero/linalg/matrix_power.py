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

def matrix_power_eig(matrix: torch.Tensor, power: float, eps: float = 1e-6) -> torch.Tensor:
    """temporary matrix^exponent using eigenvalue decomposition"""
    try:
        matrix = matrix + torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype) * eps
        eigvals, eigvecs = torch.linalg.eigh(matrix) # pylint:disable=not-callable
        inv_root_eigvals = eigvals.clamp(eigvals, min=eps).pow_(exponent=power)
        return  eigvecs @ torch.diag_embed(inv_root_eigvals) @ eigvecs.mH
    except torch.linalg.LinAlgError:
        return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)

def matrix_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """svd"""
    try:
        u, s, v = torch.svd(matrix)
        return (u @ s.pow_(power).diag() @ v.t())
    except torch.linalg.LinAlgError:
        return torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
