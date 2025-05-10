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
        inv_root_eigvals = eigvals.clamp(min=eps).pow_(exponent=power)
        return  eigvecs @ torch.diag_embed(inv_root_eigvals) @ eigvecs.mH
    except torch.linalg.LinAlgError:
        I = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        if matrix.ndim > 2: I = I.expand_as(matrix)
        return I

def matrix_power_svd(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """svd"""
    try:
        u, s, v = torch.svd(matrix)
        return (u @ s.pow_(power).diag() @ v.mT)
    except torch.linalg.LinAlgError:
        I = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        if matrix.ndim > 2: I = I.expand_as(matrix)
        return I


def inv_sqrt_2x2(A: torch.Tensor, eps: float = 1e-6, force_pd: bool=False) -> torch.Tensor:
    """Inverse square root of a possibly batched 2x2 matrix using a general formula for 2x2 matrices so that this is way faster than torch linalg."""
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]

    det = (a * d).sub_(b * c)
    trace = a + d

    if force_pd:
        # add smallest eigenvalue magnitude to diagonal to force PD
        # could also clamp eigenvalues bc there is a formula for eigenvectors
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