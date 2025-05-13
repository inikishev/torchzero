from collections.abc import Callable
from typing import overload
import torch

from ...utils import TensorList, generic_zeros_like, generic_vector_norm, generic_numel

@overload
def cg(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0_: torch.Tensor | None,
    tol: float | None,
    maxiter: int | None,
) -> torch.Tensor: ...
@overload
def cg(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0_: TensorList | None,
    tol: float | None,
    maxiter: int | None,
) -> TensorList: ...

def cg(
    A_mm: Callable,
    b: torch.Tensor | TensorList,
    x0_: torch.Tensor | TensorList | None,
    tol: float | None,
    maxiter: int | None,
):
    if maxiter is None: maxiter = generic_numel(b)
    if x0_ is None: x0_ = generic_zeros_like(b)

    x = x0_
    residual = b - A_mm(x)
    p = residual.clone() # search direction
    r_norm = generic_vector_norm(residual)
    init_norm = r_norm
    if tol is not None and r_norm < tol: return x
    k = 0

    while True:
        Ap = A_mm(p)
        step_size = (r_norm**2) / p.dot(Ap)
        x += step_size * p # Update solution
        residual -= step_size * Ap # Update residual
        new_r_norm = generic_vector_norm(residual)

        k += 1
        if tol is not None and new_r_norm <= tol * init_norm: return x
        if k >= maxiter: return x

        beta = (new_r_norm**2) / (r_norm**2)
        p = residual + beta*p
        r_norm = new_r_norm
