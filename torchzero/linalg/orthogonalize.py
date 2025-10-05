from typing import Literal

import torch

from ..utils.compile import allow_compile
from . import torch_linalg

# zeropower_via_newtonschulz5 from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# and
# https://github.com/HomebrewML/HeavyBall/blob/main/heavyball/utils.py#L452
_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012)
)

@allow_compile
def zeropower_via_newtonschulz5(G: torch.Tensor, coeffs=_NS_COEFFS) -> torch.Tensor:
    """
    Applies to last 2 dims - so usually reverse_dims should be applied to G before and after.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clip(min=torch.finfo(X.dtype).tiny * 2))

    # Perform the NS iterations
    for a,b,c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)

def zeropower_via_svd(A: torch.Tensor) -> torch.Tensor:
    """
    Applies to first 2 dims and isn't batched - rest of dimensions are flattened.
    """
    try:
        U, S, Vt = torch_linalg.svd(A, full_matrices=False, retry_float64=True) # pylint:disable=not-callable
    except torch.linalg.LinAlgError:
        U, S, Vt = torch.svd_lowrank(A, q=1, M=1e-4 * A.mean() * torch.rand_like(A))

    return  U @ Vt

def zeropower_via_eigh(A: torch.Tensor) -> torch.Tensor:
    """
    Only SPD and I need to check if I apply those to SPD because this is better than SVD.
    """
    L, Q = torch_linalg.eigh(A, retry_float64=True)
    return  Q @ Q.mH


def orthogonalize_via_qr(A: torch.Tensor):
    *_, m, n = A.shape
    T = False
    if m < n:
        T = True
        m,n = n,m
        A = A.mH

    Q = torch_linalg.qr(A, mode='reduced', retry_float64=True).Q

    if T:
        Q = Q.mH

    return Q

OrthogonalizeMethod = Literal["newtonschulz", "svd", "qr"]
def orthogonalize(A: torch.Tensor, method: OrthogonalizeMethod) -> torch.Tensor:
    if method == "newtonschulz": return zeropower_via_newtonschulz5(A)
    if method == "svd": return zeropower_via_svd(A)
    if method == "qr": return orthogonalize_via_qr(A)
    if method == "eigh": return zeropower_via_eigh(A)
    raise ValueError(method)