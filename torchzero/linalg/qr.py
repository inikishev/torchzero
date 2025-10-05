from typing import Literal
import torch
from ..utils.compile import allow_compile


# super slow
# def cholesky_qr(A):
#     """QR of (m, n) A via cholesky of (n, n) matrix"""
#     AtA = A.T @ A

#     L, _ = torch.linalg.cholesky_ex(AtA) # pylint:disable=not-callable
#     R = L.T

#     Q = torch.linalg.solve_triangular(R.T, A.T, upper=False).T # pylint:disable=not-callable
#     return Q, R

# reference - https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
@allow_compile
def _get_w_tau(R: torch.Tensor, i: int, eps: float):
    R_ii = R[...,i,i]
    R_below = R[...,i:,i]
    norm_x = torch.linalg.vector_norm(R_below, dim=-1) # pylint:disable=not-callable
    degenerate = norm_x < eps
    s = -torch.sign(R_ii)
    u1 = R_ii - s*norm_x
    u1 = torch.where(degenerate, 1, u1)
    w = R_below / u1.unsqueeze(-1)
    w[...,0] = 1
    tau = -s*u1/norm_x
    tau = torch.where(degenerate, 1, tau)
    return w, tau

@allow_compile
def _qr_householder_complete(A:torch.Tensor):
    *b,m,n = A.shape
    k = min(m,n)
    eps = torch.finfo(A.dtype).tiny * 2

    Q = torch.eye(m, dtype=A.dtype, device=A.device).expand(*b, m, m).clone() # clone because expanded dims refer to same memory
    R = A.clone()

    for i in range(k):
        w, tau = _get_w_tau(R, i, eps)

        R[..., i:,:] -= (tau*w).unsqueeze(-1) @ (w.unsqueeze(-2) @ R[..., i:,:])
        Q[..., :,i:] -= (Q[..., :,i:]@w).unsqueeze(-1) @ (tau*w).unsqueeze(-2)

    return Q, R

@allow_compile
def _qr_householder_reduced(A:torch.Tensor):
    *b,m,n = A.shape
    k = min(m,n)
    eps = torch.finfo(A.dtype).tiny * 2

    R = A.clone()

    ws:list = [None for _ in range(k)]
    taus:list = [None for _ in range(k)]

    for i in range(k):
        w, tau = _get_w_tau(R, i, eps)

        ws[i] = w
        taus[i] = tau

        if m - i > 0 :
            R[...,  i:,:] -= (tau*w).unsqueeze(-1) @ (w.unsqueeze(-2) @ R[..., i:,:])
            # Q[..., :,i:] -= (Q[..., :,i:]@w).unsqueeze(-1) @ (tau*w).unsqueeze(-2)

    R = R[..., :k, :]
    Q = torch.eye(m, k, dtype=A.dtype, device=A.device).expand(*b, m, k).clone()
    for i in range(k - 1, -1, -1):
        if m - i > 0:
            w = ws[i]
            tau = taus[i].unsqueeze(-1).unsqueeze(-1)
            Q_below = Q[..., i:, :]
            Q[..., i:, :] -= torch.linalg.multi_dot([tau * w.unsqueeze(-1), w.unsqueeze(-2), Q_below]) # pylint:disable=not-callable

    return Q, R

def qr_householder(A:torch.Tensor, mode: Literal['complete', 'reduced'] = 'reduced'):
    """an attempt at making QR decomposition for very tall and thin matrices that doesn't freeze, but it is around n_cols times slower than torch.linalg.qr, but compilation makes it faster, but it has to recompile when processing different shapes"""
    if mode == 'reduced': return _qr_householder_reduced(A)
    return _qr_householder_complete(A)