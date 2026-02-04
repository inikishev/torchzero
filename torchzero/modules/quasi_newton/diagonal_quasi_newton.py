from typing import Any, Literal
from collections.abc import Callable

import torch
from ...core import Chainable
from .quasi_newton import (
    HessianUpdateStrategy,
    _HessianUpdateStrategyDefaults,
    _InverseHessianUpdateStrategyDefaults,
)

from ..opt_utils import safe_clip


def diagonal_bfgs_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy < tol: return H

    sy_sq = safe_clip(sy**2)

    num1 = (sy + (y * H * y)) * s*s
    term1 = num1.div_(sy_sq)
    num2 = (H * y * s).add_(s * y * H)
    term2 = num2.div_(sy)
    H += term1.sub_(term2)
    return H

class DiagonalBFGS(_InverseHessianUpdateStrategyDefaults):
    """Diagonal BFGS. This is simply BFGS with only the diagonal being updated and used. It doesn't satisfy the secant equation but may still be useful."""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_bfgs_H_(H=H, s=s, y=y, tol=setting['tol'])

    def initialize_P(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)

def diagonal_sr1_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol:float):
    z = s - H*y
    denom = z.dot(y)

    z_norm = torch.linalg.norm(z) # pylint:disable=not-callable
    y_norm = torch.linalg.norm(y) # pylint:disable=not-callable

    # if y_norm*z_norm < tol: return H

    # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
    if denom.abs() <= tol * y_norm * z_norm: return H # pylint:disable=not-callable
    H += (z*z).div_(safe_clip(denom))
    return H
class DiagonalSR1(_InverseHessianUpdateStrategyDefaults):
    """Diagonal SR1. This is simply SR1 with only the diagonal being updated and used. It doesn't satisfy the secant equation but may still be useful."""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_sr1_(H=H, s=s, y=y, tol=setting['tol'])
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_sr1_(H=B, s=y, y=s, tol=setting['tol'])

    def initialize_P(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)



# Zhu M., Nazareth J. L., Wolkowicz H. The quasi-Cauchy relation and diagonal updating //SIAM Journal on Optimization. – 1999. – Т. 9. – №. 4. – С. 1192-1204.
def diagonal_qc_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = safe_clip((s**4).sum())
    num = s.dot(y) - (s*B).dot(s)
    B += s**2 * (num/denom)
    return B

class DiagonalQuasiCauchi(_HessianUpdateStrategyDefaults):
    """Diagonal quasi-cauchi method.

    Reference:
        Zhu M., Nazareth J. L., Wolkowicz H. The quasi-Cauchy relation and diagonal updating //SIAM Journal on Optimization. – 1999. – Т. 9. – №. 4. – С. 1192-1204.
    """
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_qc_B_(B=B, s=s, y=y)

    def initialize_P(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)

# Leong, Wah June, Sharareh Enshaei, and Sie Long Kek. "Diagonal quasi-Newton methods via least change updating principle with weighted Frobenius norm." Numerical Algorithms 86 (2021): 1225-1241.
def diagonal_wqc_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    E_sq = s**2 * B**2
    denom = safe_clip((s*E_sq).dot(s))
    num = s.dot(y) - (s*B).dot(s)
    B += E_sq * (num/denom)
    return B

class DiagonalWeightedQuasiCauchi(_HessianUpdateStrategyDefaults):
    """Diagonal quasi-cauchi method.

    Reference:
        Leong, Wah June, Sharareh Enshaei, and Sie Long Kek. "Diagonal quasi-Newton methods via least change updating principle with weighted Frobenius norm." Numerical Algorithms 86 (2021): 1225-1241.
    """
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_wqc_B_(B=B, s=s, y=y)

    def initialize_P(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)

def _truncate(B: torch.Tensor, lb, ub):
    return torch.where((B>lb).logical_and(B<ub), B, 1)

# Andrei, Neculai. "A diagonal quasi-Newton updating method for unconstrained optimization." Numerical Algorithms 81.2 (2019): 575-590.
def dnrtr_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = safe_clip((s**4).sum())
    num = s.dot(y) + s.dot(s) - (s*B).dot(s)
    B += s**2 * (num/denom) - 1
    return B

class DNRTR(HessianUpdateStrategy):
    """Diagonal quasi-newton method.

    Reference:
        Andrei, Neculai. "A diagonal quasi-Newton updating method for unconstrained optimization." Numerical Algorithms 81.2 (2019): 575-590.
    """
    def __init__(
        self,
        lb: float = 1e-2,
        ub: float = 1e5,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-32,
        ptol: float | None = 1e-32,
        ptol_restart: bool = False,
        gtol: float | None = 1e-32,
        restart_interval: int | None | Literal['auto'] = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(lb=lb, ub=ub)
        super().__init__(
            defaults=defaults,
            init_scale=init_scale,
            tol=tol,
            ptol=ptol,
            ptol_restart=ptol_restart,
            gtol=gtol,
            restart_interval=restart_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            concat_params=concat_params,
            inverse=False,
            inner=inner,
        )

    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_wqc_B_(B=B, s=s, y=y)

    def modify_B(self, B, state, setting):
        return _truncate(B, setting['lb'], setting['ub'])

    def initialize_P(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)

# Nosrati, Mahsa, and Keyvan Amini. "A new diagonal quasi-Newton algorithm for unconstrained optimization problems." Applications of Mathematics 69.4 (2024): 501-512.
def new_dqn_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = safe_clip((s**4).sum())
    num = s.dot(y)
    B += s**2 * (num/denom)
    return B

class NewDQN(DNRTR):
    """Diagonal quasi-newton method.

    Reference:
        Nosrati, Mahsa, and Keyvan Amini. "A new diagonal quasi-Newton algorithm for unconstrained optimization problems." Applications of Mathematics 69.4 (2024): 501-512.
    """
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return new_dqn_B_(B=B, s=s, y=y)
