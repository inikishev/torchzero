from typing import Literal
import warnings
from collections.abc import Callable

import torch
from . import torch_linalg
def matrix_power_eigh(A: torch.Tensor, power:float, abs:bool=False):
    """this is faster than SVD but only for positive semi-definite symmetric matrices
    (covariance matrices are always SPD)"""

    L, Q = torch_linalg.eigh(A, retry_float64=True) # pylint:disable=not-callable
    if abs: L.abs_()
    if power % 2 != 0: L.clip_(min = torch.finfo(A.dtype).tiny * 2)
    return (Q * L.pow_(power).unsqueeze(-2)) @ Q.mH


def matrix_power_svd(A: torch.Tensor, power: float) -> torch.Tensor:
    """for any symmetric matrix"""
    U, S, Vh = torch_linalg.svd(A, full_matrices=False, retry_float64=True) # pylint:disable=not-callable
    if power % 2 != 0: S.clip_(min = torch.finfo(A.dtype).tiny * 2)
    return (U * S.pow_(power).unsqueeze(-2)) @ Vh

MatrixPowerMethod = Literal["eigh", "eigh_abs", "svd"]
def matrix_power(A: torch.Tensor, power: float, method: MatrixPowerMethod = "eigh_abs") -> torch.Tensor:
    if method == "eigh": return matrix_power_eigh(A, power)
    if method == "eigh_abs": return matrix_power_eigh(A, power, abs=True)
    if method == "svd": return matrix_power_svd(A, power)
    raise ValueError(method)