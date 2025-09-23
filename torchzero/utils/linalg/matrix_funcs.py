import warnings
from collections.abc import Callable

import torch

def matrix_power_eigh(A: torch.Tensor, power:float, abs:bool=False):
    """this is faster than SVD but only for positive semi-definite symmetric matrices
    (covariance matrices are always SPD)"""

    try:
        L, Q = torch.linalg.eigh(A) # pylint:disable=not-callable
        if abs: L.abs_()
        if power % 2 != 0: L.clip_(min = torch.finfo(A.dtype).tiny * 2)
        return (Q * L.pow_(power).unsqueeze(-2)) @ Q.mH

    except torch.linalg.LinAlgError as e:
        dtype = A.dtype
        if dtype == torch.float64: raise e
        return matrix_power_eigh(A.to(torch.float64), power, abs=abs).to(dtype)


def matrix_power_svd(A: torch.Tensor, power: float) -> torch.Tensor:
    """for any symmetric matrix"""
    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False) # pylint:disable=not-callable
        if power % 2 != 0: S.clip_(min = torch.finfo(A.dtype).tiny * 2)
        return (U * S.pow_(power).unsqueeze(-2)) @ Vh

    except torch.linalg.LinAlgError as e:
        dtype = A.dtype
        if dtype == torch.float64: raise e
        return matrix_power_svd(A.to(torch.float64), power).to(dtype)
