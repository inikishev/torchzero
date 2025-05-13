from typing import Any, Literal
from abc import ABC, abstractmethod
from collections.abc import Mapping
import torch

from ...core import Chainable, Module, Preconditioner, TensorwisePreconditioner
from ...utils import TensorList

def _safe_dict_update_(d1_:dict, d2:dict):
    inter = set(d1_.keys()).intersection(d2.keys())
    if len(inter) > 0: raise RuntimeError(f"Duplicate keys {inter}")
    d1_.update(d2)

def _maybe_lerp_(state, key, value: torch.Tensor, beta: float | None):
    if (beta is None) or (beta == 0) or (key not in state): state[key] = value
    elif state[key].shape != value.shape: state[key] = value
    else: state[key].lerp_(value, 1-beta)

class HessianUpdateStrategy(TensorwisePreconditioner, ABC):
    def __init__(
        self,
        defaults: dict | None = None,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inverse: bool = True,
        inner: Chainable | None = None,
    ):
        if defaults is None: defaults = {}
        _safe_dict_update_(defaults, dict(init_scale=init_scale, tol=tol, scale_second=scale_second, inverse=inverse, beta=beta))
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    def update_H(self, H:torch.Tensor, s:torch.Tensor, y:torch.Tensor, p:torch.Tensor, g:torch.Tensor,
                 p_prev:torch.Tensor, g_prev:torch.Tensor, state: dict[str, Any], settings: Mapping[str, Any]) -> torch.Tensor:
        """update hessian inverse"""
        raise NotImplementedError

    def update_B(self, B:torch.Tensor, s:torch.Tensor, y:torch.Tensor, p:torch.Tensor, g:torch.Tensor,
                 p_prev:torch.Tensor, g_prev:torch.Tensor, state: dict[str, Any], settings: Mapping[str, Any]) -> torch.Tensor:
        """update hessian"""
        raise NotImplementedError

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param.view(-1); g = tensor.view(-1)
        inverse = settings['inverse']
        M_key = 'H' if inverse else 'B'
        M = state.get(M_key, None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']

        if M is None:
            M = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: M *= init_scale
            state[M_key] = M
            state['p_prev'] = p.clone()
            state['g_prev'] = g.clone()
            return

        p_prev = state['p_prev']
        g_prev = state['g_prev']
        s: torch.Tensor = p - p_prev
        y: torch.Tensor = g - g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y.abs().max() <= tol:
            return

        if step == 1 and init_scale == 'auto':
            ys = y.dot(s)
            yy = y.dot(y)
            if ys != 0 and yy != 0:
                if inverse: M *= ys/yy
                else: M *= yy/ys

        if inverse:
            _maybe_lerp_(state, 'H',
                         self.update_H(H=M, s=s, y=y, p=p, g=g, p_prev=p_prev, g_prev=g_prev, state=state, settings=settings),
                         settings['beta'])
        else:
            _maybe_lerp_(state, 'B',
                         self.update_B(B=M, s=s, y=y, p=p, g=g, p_prev=p_prev, g_prev=g_prev, state=state, settings=settings),
                         settings['beta'])

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        step = state['step'] = state.get('step', 0) + 1

        if settings['scale_second'] and step == 2:
            tensor = tensor/max(1, tensor.abs().sum()) # pyright:ignore[reportArgumentType]

        inverse = settings['inverse']
        if inverse:
            H = state['H']
            return (H @ tensor.view(-1)).view_as(tensor)

        else:
            B = state['B']
            return torch.linalg.solve(B, tensor.view(-1)).view_as(tensor) # pylint:disable=not-callable

# ----------------------------------- BFGS ----------------------------------- #
def bfgs_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    sy = torch.dot(s, y)
    if sy <= 1e-10: return H
    num1 = (sy + (y @ H @ y)) * s.outer(s)
    term1 = num1.div_(sy**2)
    num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
    term2 = num2.div_(sy)
    H += term1.sub_(term2)
    return H

class BFGS(HessianUpdateStrategy):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
            defaults=None,
            init_scale=init_scale,
            tol=tol,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )

    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return bfgs_H_(H=H, s=s, y=y)

# ------------------------------------ SR1 ----------------------------------- #
def sr1_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, eps:float):
    z = s - H@y
    denom = torch.dot(z, y)

    # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
    if denom.abs() >= eps * torch.linalg.norm(y) * torch.linalg.norm(z): # pylint:disable=not-callable
        H += torch.outer(z, z).div_(denom)
    return H

class SR1(HessianUpdateStrategy):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 1,
        eps: float = 1e-8,
        tol: float = 1e-10,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(eps=eps)
        super().__init__(
            defaults=defaults,
            init_scale=init_scale,
            tol=tol,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )

    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return sr1_H_(H=H, s=s, y=y, eps=settings['eps'])

# BFGS has defaults - init_scale = "auto" and scale_second = False
# SR1 has defaults -  init_scale = 1 and scale_second = True
# basically some methods work better with first and some with second.
# I inherit from BFGS or SR1 to avoid writing all those arguments again
# ------------------------------------ DFP ----------------------------------- #
def dfp_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    sy = torch.dot(s, y)
    if sy.abs() <= 1e-10: return H
    term1 = torch.outer(s, s).div_(sy)
    denom = torch.dot(y, H @ y) #
    if denom.abs() <= 1e-10: return H
    num = H @ torch.outer(y, y) @ H
    term2 = num.div_(denom)
    H += term1.sub_(term2)
    return H

class DFP(SR1):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return dfp_H_(H=H, s=s, y=y)


# formulas for methods below from Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
# H' = H - (Hy - S)c^T / c^T*y
# the difference is how `c` is calculated

def broyden_good_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    c = H.T @ s
    denom = c.dot(y)
    if denom.abs() <= 1e-10: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def broyden_bad_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    c = y
    denom = c.dot(y)
    if denom.abs() <= 1e-10: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def greenstadt1_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, g: torch.Tensor):
    c = g
    denom = c.dot(y)
    if denom.abs() <= 1e-10: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def greenstadt2_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor):
    c = torch.linalg.multi_dot([H,H,y]) # pylint:disable=not-callable
    denom = c.dot(y)
    if denom.abs() <= 1e-10: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

class BroydenGood(BFGS):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_good_H_(H=H, s=s, y=y)

class BroydenBad(BFGS):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_bad_H_(H=H, s=s, y=y)

class Greenstadt1(BFGS):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt1_H_(H=H, s=s, y=y, g=g)

class Greenstadt2(BFGS):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt2_H_(H=H, s=s, y=y)


def column_updating_H_(H:torch.Tensor, s:torch.Tensor, y:torch.Tensor):
    n = H.shape[0]

    j = y.abs().argmax()
    u = torch.zeros(n, device=H.device, dtype=H.dtype)
    u[j] = 1.0

    denom = y[j]
    if denom.abs() < 1e-10: return H

    Hy = H @ y.unsqueeze(1)
    num = s.unsqueeze(1) - Hy

    H[:, j] += num.squeeze() / denom
    return H

class ColumnUpdatingMethod(BFGS):
    """Lopes, V. L., & Martínez, J. M. (1995). Convergence properties of the inverse column-updating method. Optimization Methods & Software, 6(2), 127–144. from https://www.ime.unicamp.br/sites/default/files/pesquisa/relatorios/rp-1993-76.pdf"""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return column_updating_H_(H=H, s=s, y=y)

def thomas_H_(H: torch.Tensor, R:torch.Tensor, s: torch.Tensor, y: torch.Tensor):
    s_norm = torch.linalg.vector_norm(s) # pylint:disable=not-callable
    I = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
    d = (R + I * (s_norm/2)) @ s
    denom = d.dot(s)
    if denom.abs() <= 1e-10: return H, R
    R = (1 + s_norm) * ((I*s_norm).add_(R).sub_(d.outer(d).div_(denom)))

    c = H.T @ d
    denom = c.dot(y)
    if denom.abs() <= 1e-10: return H, R
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H, R

class ThomasOptimalMethod(SR1):
    """Thomas, Stephen Walter. Sequential estimation techniques for quasi-Newton algorithms. Cornell University, 1975."""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        if 'R' not in state: state['R'] = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
        H, state['R'] = thomas_H_(H=H, R=state['R'], s=s, y=y)
        return H

