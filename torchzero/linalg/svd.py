import torch

from . import torch_linalg


def tall_reduced_svd_via_eigh(A: torch.Tensor, tol: float = 0, retry_float64:bool=False):
    """
    Given a tall matrix A of size (m, n), computes U and S from the reduced SVD(A)
    using the eigendecomposition of (n, n) matrix which is faster than direct SVD when m >= n.

    This truncates small singular values that would causes nans,
    so the returned U and S can have reduced dimension ``k <= n``.

    Returns U of size ``(m, k)`` and S of size ``(k, )``.

    Args:
        A (torch.Tensor): A tall matrix of size (m, n) with m >= n.
        tol (float): Tolerance for truncating small singular values. Singular values
                     less than ``tol * max_singular_value`` will be discarded.


    """
    # if m < n, A.T A will be low rank and we can't use eigh
    m, n = A.size()
    if m < n:
        U, S, V = torch_linalg.svd(A, full_matrices=False, retry_float64=retry_float64)
        return U, S

    M = A.mH @ A # n,n

    try:
        L, Q = torch_linalg.eigh(M, retry_float64=retry_float64)
    except torch.linalg.LinAlgError:
        U, S, V = torch_linalg.svd(A, full_matrices=False, retry_float64=retry_float64)
        return U, S

    L = torch.flip(L, dims=[-1])
    Q = torch.flip(Q, dims=[-1])

    indices = L > tol * L[0] # L[0] is the max eigenvalue
    L = L[indices]
    Q = Q[:, indices]

    S = L.sqrt()
    U = (A @ Q) / S

    return U, S