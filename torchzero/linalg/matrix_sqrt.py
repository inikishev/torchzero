import torch

# sqrtmh function from https://github.com/pytorch/pytorch/issues/25481#issuecomment-1109537907
def sqrtmh(A):
    """Compute the square root of a Symmetric or Hermitian positive definite matrix or batch of matrices"""
    L, Q = torch.linalg.eigh(A)
    zero = torch.zeros((), device=L.device, dtype=L.dtype)
    threshold = L.max(-1).values * L.size(-1) * torch.finfo(L.dtype).eps
    L = L.where(L > threshold.unsqueeze(-1), zero)  # zero out small components
    return ((Q * L.sqrt().unsqueeze(-2)) @ Q.mH).real