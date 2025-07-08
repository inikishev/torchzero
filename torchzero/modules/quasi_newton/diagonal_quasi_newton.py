from collections.abc import Callable

import torch

from .quasi_newton import (
    HessianUpdateStrategy,
    _HessianUpdateStrategyDefaults,
    _InverseHessianUpdateStrategyDefaults,
    _safe_clip,
)


def _diag_Bv(self: HessianUpdateStrategy):
    B, is_inverse = self.get_B()

    if is_inverse:
        H=B
        def Hxv(v): return v/H
        return Hxv

    def Bv(v): return B*v
    return Bv

def _diag_Hv(self: HessianUpdateStrategy):
    H, is_inverse = self.get_H()

    if is_inverse:
        B=H
        def Bxv(v): return v/B
        return Bxv

    def Hv(v): return H*v
    return Hv

def diagonal_bfgs_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy < tol: return H

    sy_sq = _safe_clip(sy**2)

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

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)

def diagonal_sr1_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol:float):
    z = s - H*y
    denom = z.dot(y)

    z_norm = torch.linalg.norm(z) # pylint:disable=not-callable
    y_norm = torch.linalg.norm(y) # pylint:disable=not-callable

    # if y_norm*z_norm < tol: return H

    # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
    if denom.abs() <= tol * y_norm * z_norm: return H # pylint:disable=not-callable
    H += (z*z).div_(_safe_clip(denom))
    return H
class DiagonalSR1(_InverseHessianUpdateStrategyDefaults):
    """Diagonal SR1. This is simply SR1 with only the diagonal being updated and used. It doesn't satisfy the secant equation but may still be useful."""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_sr1_(H=H, s=s, y=y, tol=setting['tol'])
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_sr1_(H=B, s=y, y=s, tol=setting['tol'])

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)



# Zhu M., Nazareth J. L., Wolkowicz H. The quasi-Cauchy relation and diagonal updating //SIAM Journal on Optimization. – 1999. – Т. 9. – №. 4. – С. 1192-1204.
def diagonal_qc_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = _safe_clip((s**4).sum())
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

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)

# Leong, Wah June, Sharareh Enshaei, and Sie Long Kek. "Diagonal quasi-Newton methods via least change updating principle with weighted Frobenius norm." Numerical Algorithms 86 (2021): 1225-1241.
def diagonal_wqc_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    E_sq = s**2 * B**2
    denom = _safe_clip((s*E_sq).dot(s))
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

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)


# Andrei, Neculai. "A diagonal quasi-Newton updating method for unconstrained optimization." Numerical Algorithms 81.2 (2019): 575-590.
def dnrtr_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = _safe_clip((s**4).sum())
    num = s.dot(y) + s.dot(s) - (s*B).dot(s)
    B += s**2 * (num/denom) - 1
    return B

class DNRTR(_HessianUpdateStrategyDefaults):
    """Diagonal quasi-newton method.

    Reference:
        Andrei, Neculai. "A diagonal quasi-Newton updating method for unconstrained optimization." Numerical Algorithms 81.2 (2019): 575-590.
    """
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return diagonal_wqc_B_(B=B, s=s, y=y)

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)

# Nosrati, Mahsa, and Keyvan Amini. "A new diagonal quasi-Newton algorithm for unconstrained optimization problems." Applications of Mathematics 69.4 (2024): 501-512.
def new_dqn_B_(B:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    denom = _safe_clip((s**4).sum())
    num = s.dot(y)
    B += s**2 * (num/denom)
    return B

class NewDQN(_HessianUpdateStrategyDefaults):
    """Diagonal quasi-newton method.

    Reference:
        Nosrati, Mahsa, and Keyvan Amini. "A new diagonal quasi-Newton algorithm for unconstrained optimization problems." Applications of Mathematics 69.4 (2024): 501-512.
    """
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, setting):
        return new_dqn_B_(B=B, s=s, y=y)

    def _init_M(self, size:int, device, dtype, is_inverse:bool): return torch.ones(size, device=device, dtype=dtype)
    def make_Bv(self): return _diag_Bv(self)
    def make_Hv(self): return _diag_Hv(self)
