"""torch linalg with correct typing and retries in float64"""
from typing import NamedTuple

import torch


def cholesky(A: torch.Tensor, *, upper=False, retry_float64:bool=False) -> torch.Tensor:
    """A - SPD, returns lower triangular L such that ``A = L @ L.mH`` also can pass L to ``torch.cholesky_solve``"""
    try:
        return torch.linalg.cholesky(A, upper=upper) # pylint:disable=not-callable

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        return cholesky(A.to(torch.float64), upper=upper, retry_float64=False).to(dtype)


class _QRTuple(NamedTuple):
    Q: torch.Tensor
    R: torch.Tensor

def qr(A: torch.Tensor, mode='reduced', retry_float64:bool=False) -> _QRTuple:
    """A - any matrix ``(*, m, n)`` (for some reason sometimes it takes ages on some matrices)

    ### Returns (if mode = "reduced"):

    Q: ``(*, m, k)`` - orthogonal

    R: ``(*, k, n)`` - upper triangular

    where ``k = min(m,n)``
    """
    try:
        return torch.linalg.qr(A, mode=mode) # pylint:disable=not-callable

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        Q, R = qr(A.to(torch.float64), mode=mode, retry_float64=False)
        return _QRTuple(Q=Q.to(dtype), R=R.to(dtype))

def eigh(A: torch.Tensor, UPLO="L", retry_float64:bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """A - symmetric, returns ``(L, Q)``, ``A = Q @ torch.diag(L) @ Q.mH``, this is faster than SVD"""
    try:
        return torch.linalg.eigh(A, UPLO=UPLO) # pylint:disable=not-callable

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        L, Q = eigh(A.to(torch.float64), UPLO=UPLO, retry_float64=False)
        return L.to(dtype), Q.to(dtype)



class _SVDTuple(NamedTuple):
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor

def svd(A: torch.Tensor, full_matrices=True, driver=None, retry_float64:bool=False) -> _SVDTuple:
    """A - any matrix ``(*, n, m)``, but slows down if A isn't well conditioned, ``A = U @ torch.diag(S) @ Vh``

    Don't forget to set ``full_matrices=False``

    ### Returns:

    U: ``(*, m, m)`` or ``(*, m, k)`` - orthogonal

    S: ``(*, k,)`` - singular values

    V^H: ``(*, n, n)`` or ``(*, n, k)`` - orthogonal

    where ``k = min(m,n)``

    ### Drivers

    drivers are only supported on CUDA so A is moved to CUDA by this function if needed

    from docs:

    If A is well-conditioned (its condition number is not too large), or you do not mind some precision loss.

    For a general matrix: ‘gesvdj’ (Jacobi method)

    If A is tall or wide (m >> n or m << n): ‘gesvda’ (Approximate method)

    If A is not well-conditioned or precision is relevant: ‘gesvd’ (QR based)

    By default (driver= None), we call ‘gesvdj’ and, if it fails, we fallback to ‘gesvd’.
    """
    # drivers are only for CUDA
    # also the only one that doesn't freeze is ‘gesvda’
    device=None
    if driver is not None:
        device = A.device
        A = A.cuda()

    try:
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices, driver=driver) # pylint:disable=not-callable
        if device is not None:
            U = U.to(device); S = S.to(device); Vh = Vh.to(device)
        return _SVDTuple(U=U, S=S, Vh=Vh)

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        U, S, Vh = svd(A.to(torch.float64), full_matrices=full_matrices, driver=driver, retry_float64=False)
        return _SVDTuple(U=U.to(dtype), S=S.to(dtype), Vh=Vh.to(dtype))

def solve(A: torch.Tensor, B: torch.Tensor, left:bool=True, retry_float64:bool=False) -> torch.Tensor:
    """I think this uses LU"""
    try:
        return torch.linalg.solve(A, B, left=left) # pylint:disable=not-callable

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        return solve(A.to(torch.float64), B.to(torch.float64), left=left, retry_float64=False).to(dtype)

class _SolveExTuple(NamedTuple):
    result: torch.Tensor
    info: int

def solve_ex(A: torch.Tensor, B: torch.Tensor, left:bool=True, retry_float64:bool=False) -> _SolveExTuple:
    """I think this uses LU"""
    result, info = torch.linalg.solve_ex(A, B, left=left) # pylint:disable=not-callable

    if info != 0:
        if not retry_float64: return _SolveExTuple(result, info)
        dtype = A.dtype
        if dtype == torch.float64: return _SolveExTuple(result, info)
        result, info = solve_ex(A.to(torch.float64), B.to(torch.float64), retry_float64=False)
        return _SolveExTuple(result.to(dtype), info)

    return _SolveExTuple(result, info)

def inv(A: torch.Tensor, retry_float64:bool=False) -> torch.Tensor:
    try:
        return torch.linalg.inv(A) # pylint:disable=not-callable

    except torch.linalg.LinAlgError as e:
        if not retry_float64: raise e
        dtype = A.dtype
        if dtype == torch.float64: raise e
        return inv(A.to(torch.float64), retry_float64=False).to(dtype)


class _InvExTuple(NamedTuple):
    inverse: torch.Tensor
    info: int

def inv_ex(A: torch.Tensor, *, check_errors=False, retry_float64:bool=False) -> _InvExTuple:
    """this retries in float64 but on fail info will be not 0"""
    inverse, info = torch.linalg.inv_ex(A, check_errors=check_errors) # pylint:disable=not-callable

    if info != 0:
        if not retry_float64: return _InvExTuple(inverse, info)
        dtype = A.dtype
        if dtype == torch.float64: return _InvExTuple(inverse, info)
        inverse, info = inv_ex(A.to(torch.float64), retry_float64=False)
        return _InvExTuple(inverse.to(dtype), info)

    return _InvExTuple(inverse, info)