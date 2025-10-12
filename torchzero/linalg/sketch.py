import math

import torch

from .orthogonalize import orthogonalize_via_qr
from .linear_operator import LinearOperator, Dense


class Permutation(LinearOperator):
    def __init__(self, indices:torch.Tensor):
        self.indices = indices
        self.device = indices.device

    def matvec(self, x):
        return x[self.indices]

    def matmat(self, X):
        return Dense(X[:, self.indices])

def orthonormal_sketch(m, k, dtype, device, generator):
    return orthogonalize_via_qr(torch.randn(m, k, dtype=dtype, device=device, generator=generator))

def rademacher_sketch(m, k, dtype, device, generator):
    rademacher = torch.bernoulli(torch.full((m, k), 0.5, device=device, dtype=dtype), generator = generator).mul_(2).sub_(1)
    return rademacher.mul_(1 / math.sqrt(m))

def row_sketch(m, k, dtype, device, generator):
    weights = torch.ones(m, dtype=dtype, device=device)
    indices = torch.multinomial(weights, k, replacement=False, generator=generator)

    P = torch.zeros(m, k, dtype=dtype, device=device)
    P[indices, range(k)] = 1
    return P

def topk_rows_sketch(v: torch.Tensor, m, k, dtype, device):
    _, indices = torch.topk(v, k)
    P = torch.zeros(m, k, dtype=dtype, device=device)
    P[indices, range(k)] = 1
    return P
