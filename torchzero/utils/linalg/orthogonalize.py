from typing import overload
import torch
from ..utils import TensorList

@overload
def gram_schmidt(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def gram_schmidt(x: TensorList, y: TensorList) -> tuple[TensorList, TensorList]: ...
def gram_schmidt(x, y):
    """makes two orthogonal vectors, only y is changed"""
    return x, y - (x*y) / ((x*x) + 1e-8)

def orthonormal_basis_via_qr(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """make an orthonormal basis using QR

    Args:
        basis (torch.Tensor): (n, ndim).
    """
    Q, _ = torch.linalg.qr(basis.T) # pylint:disable=not-callable

    return Q.T