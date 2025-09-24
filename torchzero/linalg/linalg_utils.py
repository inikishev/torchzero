from collections.abc import Callable
import torch

def mm(
    A_mv: Callable[[torch.Tensor], torch.Tensor] | None,
    A_mm: Callable[[torch.Tensor], torch.Tensor] | None,
    X
):
    """matrix-matrix when either mv or mm is given"""
    if A_mm is not None: return A_mm(X)
    assert A_mv is not None
    return torch.stack([A_mv(col) for col in X.unbind(-1)], -1) # rank matvecs


