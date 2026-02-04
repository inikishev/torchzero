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

# CODE FROM https://github.com/HomebrewML/HeavyBall/blob/main/heavyball/utils.py:

## Based on https://arxiv.org/pdf/2505.16932v3
# and https://github.com/NoahAmsel/PolarExpress/blob/5454910920ca8c65afda28820cdf9e49b9436ed0/polar_express.py#L69-L82
# and https://github.com/thinking-machines-lab/manifolds/blob/89dcae50f01af59f1e0570289474da3a2ecaa60b/src/msign.py#L47
#
# under the MIT License
# Coefficients are from https://arxiv.org/pdf/2505.16932v3
ABC_LIST: list[tuple[float, float, float]] = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]

# safety factor for numerical stability (but exclude last polynomial)
ABC_LIST_STABLE: list[tuple[float, float, float]] = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in ABC_LIST[:-1]
] + [ABC_LIST[-1]]


def msign(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Polar Express algorithm for the matrix sign function:
    https://arxiv.org/abs/2505.16932
    """
    assert G.ndim >= 2
    should_transpose: bool = G.size(-2) > G.size(-1)

    x = G
    if should_transpose:
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + eps)

    for step in range(steps):
        a, b, c = ABC_LIST_STABLE[step] if step < len(ABC_LIST_STABLE) else ABC_LIST_STABLE[-1]
        s = x @ x.mT
        # goal is to compute x = a x + b S x + c S^2 x
        # we can break this up into: x = (a I + (b I + c S) S) x
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if should_transpose:
        x = x.mT
    return x.float()


###### END

OrthogonalizeMethod = Literal["newtonschulz", "ns5", "polar_express", "svd", "qr", "eigh"]
def orthogonalize(A: torch.Tensor, method: OrthogonalizeMethod) -> torch.Tensor:
    if method in ("newtonschulz", "ns5"): return zeropower_via_newtonschulz5(A)
    if method == "polar_express": return msign(A)
    if method == "svd": return zeropower_via_svd(A)
    if method == "qr": return orthogonalize_via_qr(A)
    if method == "eigh": return zeropower_via_eigh(A)
    raise ValueError(method)