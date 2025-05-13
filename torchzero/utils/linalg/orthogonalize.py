from typing import overload
import torch
from ..tensorlist import TensorList

@overload
def gram_schmidt(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def gram_schmidt(x: TensorList, y: TensorList) -> tuple[TensorList, TensorList]: ...
def gram_schmidt(x, y):
    """makes two orthogonal vectors, only y is changed"""
    return x, y - (x*y) / ((x*x) + 1e-8)
