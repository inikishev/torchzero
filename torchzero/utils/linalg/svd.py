import torch

# projected svd
# adapted from https://github.com/smortezavi/Randomized_SVD_GPU
def randomized_svd(M: torch.Tensor, k: int, driver=None):
    *_, m, n = M.shape
    transpose = False
    if m < n:
        transpose = True
        M = M.mT
        m,n = n,m

    rand_matrix = torch.randn(size=(n, k), device=M.device, dtype=M.dtype)
    Q, _ = torch.linalg.qr(M @ rand_matrix, mode='reduced') # pylint:disable=not-callable
    smaller_matrix = Q.mT @ M
    U_hat, s, V = torch.linalg.svd(smaller_matrix, driver=driver, full_matrices=False) # pylint:disable=not-callable
    U = Q @ U_hat

    if transpose: return V.mT, s, U.mT
    return U, s, V