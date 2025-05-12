import torch

def randomized_svd(M: torch.Tensor, k: int):
    *_, m, n = M.shape
    transpose = False
    if m < n:
        transpose = True
        M = M.mT
        *_, m, n = M.shape

    rand_matrix = torch.randn(size=(n, k), device=M.device, dtype=M.dtype)
    Q, _ = torch.linalg.qr(M @ rand_matrix, mode='reduced') # pylint:disable=not-callable
    smaller_matrix = Q.mT @ M
    U_hat, s, V = torch.linalg.svd(smaller_matrix, full_matrices=False) # pylint:disable=not-callable
    U = Q @ U_hat

    if transpose: return V.mT, s, U.mT
    return U, s, V