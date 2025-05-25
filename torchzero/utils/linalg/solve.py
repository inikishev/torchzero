from collections.abc import Callable
from typing import overload
import torch

from .. import TensorList, generic_zeros_like, generic_vector_norm, generic_numel, generic_randn_like, generic_eq

@overload
def cg(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0_: torch.Tensor | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float = 0,
) -> torch.Tensor: ...
@overload
def cg(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0_: TensorList | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
) -> TensorList: ...

def cg(
    A_mm: Callable,
    b: torch.Tensor | TensorList,
    x0_: torch.Tensor | TensorList | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
):
    def A_mm_reg(x): # A_mm with regularization
        Ax = A_mm(x)
        if not generic_eq(reg, 0): Ax += x*reg
        return Ax

    if maxiter is None: maxiter = generic_numel(b)
    if x0_ is None: x0_ = generic_zeros_like(b)

    x = x0_
    residual = b - A_mm_reg(x)
    p = residual.clone() # search direction
    r_norm = generic_vector_norm(residual)
    init_norm = r_norm
    if tol is not None and r_norm < tol: return x
    k = 0

    while True:
        Ap = A_mm_reg(p)
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


# https://arxiv.org/pdf/2110.02820 algorithm 2.1 apparently supposed to be diabolical
def nystrom_approximation(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    ndim: int,
    rank: int,
    device,
    dtype = torch.float32,
    generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    omega = torch.randn((ndim, rank), device=device, dtype=dtype, generator=generator) # Gaussian test matrix
    omega, _ = torch.linalg.qr(omega) # Thin QR decomposition # pylint:disable=not-callable

    # Y = AΩ
    Y = torch.stack([A_mm(col) for col in omega.unbind(-1)], -1) # rank matvecs
    v = torch.finfo(dtype).eps * torch.linalg.matrix_norm(Y, ord='fro') # Compute shift # pylint:disable=not-callable
    Yv = Y + v*omega # Shift for stability
    C = torch.linalg.cholesky_ex(omega.mT @ Yv)[0] # pylint:disable=not-callable
    B = torch.linalg.solve_triangular(C, Yv.mT, upper=False, unitriangular=False).mT # pylint:disable=not-callable
    U, S, _ = torch.linalg.svd(B, full_matrices=False) # pylint:disable=not-callable
    lambd = (S.pow(2) - v).clip(min=0) #Remove shift, compute eigs
    return U, lambd

# this one works worse
def nystrom_sketch_and_solve(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    rank: int,
    reg: float = 1e-3,
    generator=None,
) -> torch.Tensor:
    U, lambd = nystrom_approximation(
        A_mm=A_mm,
        ndim=b.size(-1),
        rank=rank,
        device=b.device,
        dtype=b.dtype,
        generator=generator,
    )
    b = b.unsqueeze(-1)
    lambd += reg
    # x = (A + μI)⁻¹ b
    # (A + μI)⁻¹ = U(Λ + μI)⁻¹Uᵀ + (1/μ)(b - UUᵀ)
    # x = U(Λ + μI)⁻¹Uᵀb + (1/μ)(b - UUᵀb)
    Uᵀb = U.T @ b
    term1 = U @ ((1/lambd).unsqueeze(-1) * Uᵀb)
    term2 = (1.0 / reg) * (b - U @ Uᵀb)
    return (term1 + term2).squeeze(-1)

# this one is insane
def nystrom_pcg(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    sketch_size: int,
    reg: float = 1e-6,
    x0_: torch.Tensor | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    generator=None,
) -> torch.Tensor:
    U, lambd = nystrom_approximation(
        A_mm=A_mm,
        ndim=b.size(-1),
        rank=sketch_size,
        device=b.device,
        dtype=b.dtype,
        generator=generator,
    )
    lambd += reg

    def A_mm_reg(x): # A_mm with regularization
        Ax = A_mm(x)
        if reg != 0: Ax += x*reg
        return Ax

    if maxiter is None: maxiter = b.numel()
    if x0_ is None: x0_ = torch.zeros_like(b)

    x = x0_
    residual = b - A_mm_reg(x)
    # z0 = P⁻¹ r0
    term1 = lambd[...,-1] * U * (1/lambd.unsqueeze(-2)) @ U.mT
    term2 = torch.eye(U.size(-2), device=U.device,dtype=U.dtype) - U@U.mT
    P_inv = term1 + term2
    z = P_inv @ residual
    p = z.clone() # search direction

    init_norm = torch.linalg.vector_norm(residual) # pylint:disable=not-callable
    if tol is not None and init_norm < tol: return x
    k = 0
    while True:
        Ap = A_mm_reg(p)
        rz = residual.dot(z)
        step_size = rz / p.dot(Ap)
        x += step_size * p
        residual -= step_size * Ap

        k += 1
        if tol is not None and torch.linalg.vector_norm(residual) <= tol * init_norm: return x # pylint:disable=not-callable
        if k >= maxiter: return x

        z = P_inv @ residual
        beta = residual.dot(z) / rz
        p = z + p*beta

