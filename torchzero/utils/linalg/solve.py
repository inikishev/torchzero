# pyright: reportArgumentType=false
import math
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple, overload

import torch

from .. import (
    TensorList,
    generic_eq,
    generic_finfo_tiny,
    generic_numel,
    generic_vector_norm,
    generic_zeros_like,
)


def _make_A_mm_reg(A_mm: Callable, reg):
    def A_mm_reg(x): # A_mm with regularization
        Ax = A_mm(x)
        if not generic_eq(reg, 0): Ax += x*reg
        return Ax
    return A_mm_reg

def _identity(x): return x


# https://arxiv.org/pdf/2110.02820
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
    eps = torch.finfo(b.dtype).tiny * 2
    if tol is None: tol = eps

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
    if init_norm < tol: return x
    k = 0
    while True:
        Ap = A_mm_reg(p)
        rz = residual.dot(z)
        step_size = rz / p.dot(Ap)
        x += step_size * p
        residual -= step_size * Ap

        k += 1
        if torch.linalg.vector_norm(residual) <= tol * init_norm: return x # pylint:disable=not-callable
        if k >= maxiter: return x

        z = P_inv @ residual
        beta = residual.dot(z) / rz
        p = z + p*beta


def _safe_clip(x: torch.Tensor):
    """makes sure scalar tensor x is not smaller than tiny"""
    assert x.numel() == 1, x.shape
    eps = torch.finfo(x.dtype).tiny * 2
    if x.abs() < eps: return x.new_full(x.size(), eps).copysign(x)
    return x

def _trust_tau(x,d,trust_radius):
    xx = x.dot(x)
    xd = x.dot(d)
    dd = _safe_clip(d.dot(d))

    rad = (xd**2 - dd * (xx - trust_radius**2)).clip(min=0).sqrt()
    tau = (-xd + rad) / dd

    return x + tau * d


class CG:
    """Conjugate gradient method.

    Args:
        A_mm (Callable[[torch.Tensor], torch.Tensor] | torch.Tensor): Callable that returns matvec ``Ax``.
        b (torch.Tensor): right hand side
        x0 (torch.Tensor | None, optional): initial guess, defaults to zeros. Defaults to None.
        tol (float | None, optional): tolerance for convergence. Defaults to 1e-8.
        maxiter (int | None, optional):
            maximum number of iterations, if None sets to number of dimensions. Defaults to None.
        reg (float, optional): regularization. Defaults to 0.
        trust_radius (float | None, optional):
            CG is terminated whenever solution exceeds trust region, returning a solution modified to be within it. Defaults to None.
        npc_terminate (bool, optional):
            whether to terminate CG whenever negative curavture is detected. Defaults to False.
        miniter (int, optional):
            minimal number of iterations even if tolerance is satisfied, this ensures some progress
            is always made.
        history_size (int, optional):
            number of past iterations to store, to re-use them when trust radius is decreased.
        P_mm (Callable | torch.Tensor | None, optional):
            Callable that returns inverse preconditioner times vector. Defaults to None.
    """
    def __init__(
        self,
        A_mm: Callable,
        b: torch.Tensor | TensorList,
        x0: torch.Tensor | TensorList | None = None,
        tol: float | None = 1e-4,
        maxiter: int | None = None,
        reg: float = 0,
        trust_radius: float | None = None,
        npc_terminate: bool=False,
        miniter: int = 0,
        history_size: int = 0,
        P_mm: Callable | None = None,
):
        # --------------------------------- set attrs -------------------------------- #
        self.A_mm = _make_A_mm_reg(A_mm, reg)
        self.b = b
        if tol is None: tol = generic_finfo_tiny(b) * 2
        self.tol = tol
        self.eps = generic_finfo_tiny(b) * 2
        if maxiter is None: maxiter = generic_numel(b)
        self.maxiter = maxiter
        self.miniter = miniter
        self.trust_radius = trust_radius
        self.npc_terminate = npc_terminate
        self.P_mm = P_mm if P_mm is not None else _identity

        if history_size > 0:
            self.history = deque(maxlen = history_size)
            """history of (x, x_norm, d)"""
        else:
            self.history = None

        # -------------------------------- initialize -------------------------------- #

        self.iter = 0

        if x0 is None:
            self.x = generic_zeros_like(b)
            self.r = b
        else:
            self.x = x0
            self.r = b - A_mm(self.x)

        self.z = self.P_mm(self.r)
        self.d = self.z

        if self.history is not None:
            self.history.append((self.x, generic_vector_norm(self.x), self.d))

    def step(self) -> tuple[Any, bool]:
        """returns ``(solution, should_terminate)``"""
        x, b, d, r, z = self.x, self.b, self.d, self.r, self.z

        if self.iter >= self.maxiter:
            return x, True

        Ad = self.A_mm(d)
        dAd = d.dot(Ad)

        # check negative curvature
        if dAd <= self.eps:
            if self.trust_radius is not None: return _trust_tau(x, d, self.trust_radius), True
            if self.iter == 0: return b * (b.dot(b) / dAd).abs(), True
            if self.npc_terminate: return x, True

        rz = r.dot(z)
        alpha = rz / dAd
        x_next = x + alpha * d

        # check if the step exceeds the trust-region boundary
        x_next_norm = None
        if self.trust_radius is not None:
            x_next_norm = generic_vector_norm(x_next)
            if x_next_norm >= self.trust_radius:
                return _trust_tau(x, d, self.trust_radius), True

        # update step, residual and direction
        r_next = r - alpha * Ad

        # check if r is sufficiently small
        if self.iter >= self.miniter and generic_vector_norm(r_next) < self.tol:
            return x_next, True

        # update d, r, z
        z_next = self.P_mm(r_next)
        beta = r_next.dot(z_next) / rz

        self.d = z_next + beta * d
        self.x = x_next
        self.r = r_next
        self.z = z_next

        # update history
        if self.history is not None:
            if x_next_norm is None: x_next_norm = generic_vector_norm(x_next)
            self.history.append((self.x, x_next_norm, self.d))

        self.iter += 1
        return x, False


    def solve(self):
        # return initial guess if it is good enough
        if self.miniter < 1 and generic_vector_norm(self.r) < self.tol:
            return self.x

        should_terminate = False
        sol = None

        while not should_terminate:
            sol, should_terminate = self.step()

        assert sol is not None
        return sol

def find_within_trust_radius(history, trust_radius: float):
    """find first ``x`` in history that exceeds trust radius, if no such ``x`` exists, returns ``None``"""
    for x, x_norm, d in reversed(tuple(history)):
        if x_norm <= trust_radius:
            return _trust_tau(x, d, trust_radius)
    return None

class _TensorSolution(NamedTuple):
    x: torch.Tensor
    solver: CG

class _TensorListSolution(NamedTuple):
    x: TensorList
    solver: CG


@overload
def cg(
    A_mm: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float | None = 1e-8,
    maxiter: int | None = None,
    reg: float = 0,
    trust_radius: float | None = None,
    npc_terminate: bool = False,
    miniter: int = 0,
    history_size: int = 0,
    P_mm: Callable[[torch.Tensor], torch.Tensor] | None = None
) -> _TensorSolution: ...
@overload
def cg(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0: TensorList | None = None,
    tol: float | None = 1e-8,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
    trust_radius: float | None = None,
    npc_terminate: bool=False,
    miniter: int = 0,
    history_size: int = 0,
    P_mm: Callable[[TensorList], TensorList] | None = None
) -> _TensorListSolution: ...
def cg(
    A_mm: Callable,
    b: torch.Tensor | TensorList,
    x0: torch.Tensor | TensorList | None = None,
    tol: float | None = 1e-8,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
    trust_radius: float | None = None,
    npc_terminate: bool = False,
    miniter: int = 0,
    history_size:int = 0,
    P_mm: Callable | None = None
):
    solver = CG(
        A_mm=A_mm,
        b=b,
        x0=x0,
        tol=tol,
        maxiter=maxiter,
        reg=reg,
        trust_radius=trust_radius,
        npc_terminate=npc_terminate,
        miniter=miniter,
        history_size=history_size,
        P_mm=P_mm,
    )

    x = solver.solve()

    if isinstance(b, torch.Tensor):
        return _TensorSolution(x, solver)

    return _TensorListSolution(x, solver)


# Liu, Yang, and Fred Roosta. "MINRES: From negative curvature detection to monotonicity properties." SIAM Journal on Optimization 32.4 (2022): 2636-2661.
@overload
def minres(
    A_mm: Callable[[torch.Tensor], torch.Tensor] | torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float = 0,
    npc_terminate: bool=True,
    trust_radius: float | None = None,
) -> torch.Tensor: ...
@overload
def minres(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0: TensorList | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
    npc_terminate: bool=True,
    trust_radius: float | None = None,
) -> TensorList: ...
def minres(
    A_mm,
    b,
    x0: torch.Tensor | TensorList | None = None,
    tol: float | None = 1e-4,
    maxiter: int | None = None,
    reg: float | list[float] | tuple[float] = 0,
    npc_terminate: bool=True,
    trust_radius: float | None = None, #trust region is experimental
):
    A_mm_reg = _make_A_mm_reg(A_mm, reg)
    eps = math.sqrt(generic_finfo_tiny(b) * 2)
    if tol is None: tol = eps

    if maxiter is None: maxiter = generic_numel(b)
    if x0 is None:
        R = b
        x0 = generic_zeros_like(b)
    else:
        R = b - A_mm_reg(x0)

    X: Any = x0
    beta = b_norm = generic_vector_norm(b)
    if b_norm < eps**2:
        return generic_zeros_like(b)


    V = b / beta
    V_prev = generic_zeros_like(b)
    D = generic_zeros_like(b)
    D_prev = generic_zeros_like(b)

    c = -1
    phi = tau = beta
    s = delta1 = e = 0


    for _ in range(maxiter):

        P = A_mm_reg(V)
        alpha = V.dot(P)
        P -= beta*V_prev
        P -= alpha*V
        beta = generic_vector_norm(P)

        delta2 = c*delta1 + s*alpha
        gamma1 = s*delta1 - c*alpha
        e_next = s*beta
        delta1 = -c*beta

        cgamma1 = c*gamma1
        if trust_radius is not None and cgamma1 >= 0:
            if npc_terminate: return _trust_tau(X, R, trust_radius)
            return _trust_tau(X, D, trust_radius)

        if npc_terminate and cgamma1 >= 0:
            return R

        gamma2 = (gamma1**2 + beta**2)**(1/2)

        if abs(gamma2) <= eps: # singular system
            # c=0; s=1; tau=0
            if trust_radius is None: return X
            return _trust_tau(X, D, trust_radius)

        c = gamma1 / gamma2
        s = beta/gamma2
        tau = c*phi
        phi = s*phi

        D_prev = D
        D = (V - delta2*D - e*D_prev) / gamma2
        e = e_next
        X = X + tau*D

        if trust_radius is not None:
            if generic_vector_norm(X) > trust_radius:
                return _trust_tau(X, D, trust_radius)

        if (abs(beta) < eps) or (phi / b_norm <= tol):
            # R = zeros(R)
            return X

        V_prev = V
        V = P/beta
        R = s**2*R - phi*c*V

    return X