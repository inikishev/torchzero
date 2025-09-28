import math
from typing import Literal, Protocol, overload

import torch

from ...utils import TensorList
from ...linalg.linear_operator import DenseInverse, LinearOperator
from ..opt_utils import safe_clip


class DampingStrategy(Protocol):
    def __call__(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
        g: torch.Tensor,
        H: LinearOperator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return s, y

def _sy_Hs_sHs(s:torch.Tensor, y:torch.Tensor, H:LinearOperator):
    if isinstance(H, DenseInverse):
        Hs = H.solve(y)
        sHs = y.dot(Hs)
    else:
        Hs = H.matvec(s)
        sHs = s.dot(Hs)

    return s.dot(y), Hs, sHs



def powell_damping(s:torch.Tensor, y:torch.Tensor, g:torch.Tensor, H:LinearOperator, u=0.2):
    # here H is hessian! not the inverse

    sy, Hs, sHs = _sy_Hs_sHs(s, y, H)
    if sy < u*sHs:
        phi = ((1-u) * sHs) / safe_clip((sHs - sy))
        s = phi * s + (1 - phi) * Hs

    return s, y

def double_damping(s:torch.Tensor, y:torch.Tensor, g:torch.Tensor, H:LinearOperator, u1=0.2, u2=1/3):
    # Goldfarb, Donald, Yi Ren, and Achraf Bahamou. "Practical quasi-newton methods for training deep neural networks." Advances in Neural Information Processing Systems 33 (2020): 2386-2396.

    # Powell’s damping on H
    sy, Hs, sHs = _sy_Hs_sHs(s, y, H)
    if sy < u1*sHs:
        phi = ((1-u1) * sHs) / safe_clip(sHs - sy)
        s = phi * s + (1 - phi) * Hs

    # Powell’s damping with B = I
    sy = s.dot(y)
    ss = s.dot(s)

    if sy < u2*ss:
        phi = ((1-u2) * ss) / safe_clip(ss - sy)
        y = phi * y + (1 - phi) * s

    return s, y



_DAMPING_KEYS = Literal["powell", "double"]
_DAMPING_STRATEGIES: dict[_DAMPING_KEYS, DampingStrategy] = {
    "powell": powell_damping,
    "double": double_damping,
}


DampingStrategyType = _DAMPING_KEYS | DampingStrategy | None

@overload
def apply_damping(
    strategy: DampingStrategyType,
    s: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    H: LinearOperator,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@overload
def apply_damping(
    strategy: DampingStrategyType,
    s: TensorList,
    y: TensorList,
    g: TensorList,
    H: LinearOperator,
) -> tuple[TensorList, TensorList]: ...
def apply_damping(
    strategy: DampingStrategyType,
    s,
    y,
    g,
    H: LinearOperator,
):
    if strategy is None: return s, y
    if isinstance(strategy, str): strategy = _DAMPING_STRATEGIES[strategy]

    if isinstance(s, TensorList):
        assert isinstance(y, TensorList) and isinstance(g, TensorList)
        s_vec, y_vec = strategy(s.to_vec(), y.to_vec(), g.to_vec(), H)
        return s.from_vec(s_vec), y.from_vec(y_vec)

    assert isinstance(y, torch.Tensor) and isinstance(g, torch.Tensor)
    return strategy(s, y, g, H)