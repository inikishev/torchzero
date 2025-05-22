"""Use BFGS or maybe SR1."""
from typing import Any, Literal
from abc import ABC, abstractmethod
from collections.abc import Mapping
import torch

from ...core import Chainable, Module, Preconditioner, TensorwisePreconditioner
from ...utils import TensorList, set_storage_

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
        tol_reset: bool = True,
        reset_interval: int | None = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inverse: bool = True,
        inner: Chainable | None = None,
    ):
        if defaults is None: defaults = {}
        _safe_dict_update_(defaults, dict(init_scale=init_scale, tol=tol, tol_reset=tol_reset, scale_second=scale_second, inverse=inverse, beta=beta, reset_interval=reset_interval))
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    def _get_init_scale(self,s:torch.Tensor,y:torch.Tensor) -> torch.Tensor | float:
        """returns multiplier to H or B"""
        ys = y.dot(s)
        yy = y.dot(y)
        if ys != 0 and yy != 0: return yy/ys
        return 1

    def _reset_M_(self, M: torch.Tensor, s:torch.Tensor,y:torch.Tensor,inverse:bool, init_scale: Any):
        set_storage_(M, torch.eye(M.size(-1), device=M.device, dtype=M.dtype))
        if init_scale == 'auto': init_scale = self._get_init_scale(s,y)
        if init_scale >= 1:
            if inverse: M /= init_scale
            else: M *= init_scale

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
        tol_reset = settings['tol_reset']
        reset_interval = settings['reset_interval']

        if M is None:
            M = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1:
                if inverse: M /= init_scale
                else: M *= init_scale

            state[M_key] = M
            state['p_prev'] = p.clone()
            state['g_prev'] = g.clone()
            return

        p_prev = state['p_prev']
        g_prev = state['g_prev']
        s: torch.Tensor = p - p_prev
        y: torch.Tensor = g - g_prev
        state['p_prev'].copy_(p)
        state['g_prev'].copy_(g)


        if reset_interval is not None and step % reset_interval == 0:
            self._reset_M_(M, s, y, inverse, init_scale)
            return

        # tolerance on gradient difference to avoid exploding after converging
        if y.abs().max() <= tol:
            # reset history
            if tol_reset: self._reset_M_(M, s, y, inverse, init_scale)
            return

        if step == 1 and init_scale == 'auto':
            if inverse: M /= self._get_init_scale(s,y)
            else: M *= self._get_init_scale(s,y)

        beta = settings['beta']
        if beta is not None and beta != 0: M = M.clone() # because all of them update it in-place

        if inverse:
            H_new = self.update_H(H=M, s=s, y=y, p=p, g=g, p_prev=p_prev, g_prev=g_prev, state=state, settings=settings)
            _maybe_lerp_(state, 'H', H_new, beta)

        else:
            B_new = self.update_B(B=M, s=s, y=y, p=p, g=g, p_prev=p_prev, g_prev=g_prev, state=state, settings=settings)
            _maybe_lerp_(state, 'B', B_new, beta)

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        step = state['step'] = state.get('step', 0) + 1

        if settings['scale_second'] and step == 2:
            s = max(1, tensor.abs().sum()) # pyright:ignore[reportArgumentType]
            if s < settings['tol']: tensor = tensor/s

        inverse = settings['inverse']
        if inverse:
            H = state['H']
            return (H @ tensor.view(-1)).view_as(tensor)

        B = state['B']

        return torch.linalg.solve_ex(B, tensor.view(-1))[0].view_as(tensor) # pylint:disable=not-callable

# to avoid typing all arguments for each method
class QuasiNewtonH(HessianUpdateStrategy):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        tol_reset: bool = True,
        reset_interval: int | None = None,
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
            tol_reset=tol_reset,
            reset_interval=reset_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )
# ----------------------------------- BFGS ----------------------------------- #
def bfgs_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = torch.dot(s, y)
    if sy <= tol: return H # don't reset H in this case
    num1 = (sy + (y @ H @ y)) * s.outer(s)
    term1 = num1.div_(sy**2)
    num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
    term2 = num2.div_(sy)
    H += term1.sub_(term2)
    return H

class BFGS(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return bfgs_H_(H=H, s=s, y=y, tol=settings['tol'])

# ------------------------------------ SR1 ----------------------------------- #
def sr1_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol:float):
    z = s - H@y
    denom = torch.dot(z, y)

    z_norm = torch.linalg.norm(z) # pylint:disable=not-callable
    y_norm = torch.linalg.norm(y) # pylint:disable=not-callable

    if y_norm*z_norm < tol: return H

    # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
    if denom.abs() <= tol * y_norm * z_norm: return H # pylint:disable=not-callable
    H += torch.outer(z, z).div_(denom)
    return H

class SR1(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return sr1_H_(H=H, s=s, y=y, tol=settings['tol'])

# BFGS has defaults - init_scale = "auto" and scale_second = False
# SR1 has defaults -  init_scale = 1 and scale_second = True
# basically some methods work better with first and some with second.
# I inherit from BFGS or SR1 to avoid writing all those arguments again
# ------------------------------------ DFP ----------------------------------- #
def dfp_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = torch.dot(s, y)
    if sy.abs() <= tol: return H
    term1 = torch.outer(s, s).div_(sy)
    denom = torch.dot(y, H @ y) #
    if denom.abs() <= tol: return H
    num = H @ torch.outer(y, y) @ H
    term2 = num.div_(denom)
    H += term1.sub_(term2)
    return H

class DFP(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return dfp_H_(H=H, s=s, y=y, tol=settings['tol'])


# formulas for methods below from Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
# H' = H - (Hy - S)c^T / c^T*y
# the difference is how `c` is calculated

def broyden_good_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    c = H.T @ s
    denom = c.dot(y)
    if denom.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def broyden_bad_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    c = y
    denom = c.dot(y)
    if denom.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def greenstadt1_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, g_prev: torch.Tensor, tol: float):
    c = g_prev
    denom = c.dot(y)
    if denom.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

def greenstadt2_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    c = torch.linalg.multi_dot([H,H,y]) # pylint:disable=not-callable
    denom = c.dot(y)
    if denom.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H

class BroydenGood(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_good_H_(H=H, s=s, y=y, tol=settings['tol'])

class BroydenBad(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_bad_H_(H=H, s=s, y=y, tol=settings['tol'])

class Greenstadt1(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt1_H_(H=H, s=s, y=y, g_prev=g_prev, tol=settings['tol'])

class Greenstadt2(QuasiNewtonH):
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt2_H_(H=H, s=s, y=y, tol=settings['tol'])


def column_updating_H_(H:torch.Tensor, s:torch.Tensor, y:torch.Tensor, tol:float):
    n = H.shape[0]

    j = y.abs().argmax()
    u = torch.zeros(n, device=H.device, dtype=H.dtype)
    u[j] = 1.0

    denom = y[j]
    if denom.abs() < tol: return H

    Hy = H @ y.unsqueeze(1)
    num = s.unsqueeze(1) - Hy

    H[:, j] += num.squeeze() / denom
    return H

class ColumnUpdatingMethod(QuasiNewtonH):
    """Lopes, V. L., & Martínez, J. M. (1995). Convergence properties of the inverse column-updating method. Optimization Methods & Software, 6(2), 127–144. from https://www.ime.unicamp.br/sites/default/files/pesquisa/relatorios/rp-1993-76.pdf"""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return column_updating_H_(H=H, s=s, y=y, tol=settings['tol'])

def thomas_H_(H: torch.Tensor, R:torch.Tensor, s: torch.Tensor, y: torch.Tensor, tol:float):
    s_norm = torch.linalg.vector_norm(s) # pylint:disable=not-callable
    I = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
    d = (R + I * (s_norm/2)) @ s
    denom = d.dot(s)
    if denom.abs() <= tol: return H, R
    R = (1 + s_norm) * ((I*s_norm).add_(R).sub_(d.outer(d).div_(denom)))

    c = H.T @ d
    denom = c.dot(y)
    if denom.abs() <= tol: return H, R
    num = (H@y).sub_(s).outer(c)
    H -= num/denom
    return H, R

class ThomasOptimalMethod(QuasiNewtonH):
    """Thomas, Stephen Walter. Sequential estimation techniques for quasi-Newton algorithms. Cornell University, 1975."""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        if 'R' not in state: state['R'] = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
        H, state['R'] = thomas_H_(H=H, R=state['R'], s=s, y=y, tol=settings['tol'])
        return H

# ------------------------ powell's symmetric broyden ------------------------ #
def psb_B_(B: torch.Tensor, s: torch.Tensor, y: torch.Tensor, tol:float):
    y_Bs = y - B@s
    ss = s.dot(s)
    if ss.abs() < tol: return B
    num1 = y_Bs.outer(s).add_(s.outer(y_Bs))
    term1 = num1.div_(ss)
    term2 = s.outer(s).mul_(y_Bs.dot(s)/(ss**2))
    B += term1.sub_(term2)
    return B

class PSB(HessianUpdateStrategy):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 'auto',
        tol: float = 1e-10,
        tol_reset: bool = True,
        reset_interval: int | None = None,
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
            tol_reset=tol_reset,
            reset_interval=reset_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=False,
            inner=inner,
        )

    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, settings):
        return psb_B_(B=B, s=s, y=y, tol=settings['tol'])

def pearson2_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy.abs() <= tol: return H
    num = (s - H@y).outer(s)
    H += num.div_(sy)
    return H

class Pearson2(QuasiNewtonH):
    """finally found a reference in https://www.recotechnologies.com/~beigi/ps/asme-jdsmc-93-2.pdf"""
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return pearson2_H_(H=H, s=s, y=y, tol=settings['tol'])

# Oren, S. S., & Spedicato, E. (1976). Optimal conditioning of self-scaling variable metric algorithms. Mathematical programming, 10(1), 70-90.
def ssvm_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, g:torch.Tensor, switch: tuple[float,float] | Literal[1,2,3,4], tol: float):
    # in notation p is s, q is y, H is D
    # another p is lr
    # omega (o) = sy
    # tau (t) = yHy
    # epsilon = p'D^-1 p
    # however p.12 says eps = gs / gHy

    Hy = H@y
    gHy = g.dot(Hy)
    yHy = y.dot(Hy)
    sy = s.dot(y)
    if sy < tol: return H
    if yHy.abs() < tol: return H
    if gHy.abs() < tol: return H

    v_mul = yHy.sqrt()
    v_term1 = s/sy
    v_term2 = Hy/yHy
    v = (v_term1.sub_(v_term2)).mul_(v_mul)
    gs = g.dot(s)

    if isinstance(switch, tuple): phi, theta = switch
    else:
        o = sy
        t = yHy
        e = gs / gHy
        if switch in (1, 3):
            if e/o <= 1:
                if o.abs() <= tol: return H
                phi = e/o
                theta = 0
            elif o/t >= 1:
                if t.abs() <= tol: return H
                phi = o/t
                theta = 1
            else:
                phi = 1
                denom = e*t - o**2
                if denom.abs() <= tol: return H
                if switch == 1: theta = o * (e - o) / denom
                else: theta = o * (t - o) / denom

        elif switch == 2:
            if t.abs() <= tol or o.abs() <= tol or e.abs() <= tol: return H
            phi = (e / t) ** 0.5
            theta = 1 / (1 + (t*e / o**2)**0.5)

        elif switch == 4:
            if t.abs() <= tol: return H
            phi = e/t
            theta = 1/2

        else: raise ValueError(switch)


    u = phi * (gs/gHy) + (1 - phi) * (sy/yHy)
    term1 = (H @ y.outer(y) @ H).div_(yHy)
    term2 = v.outer(v).mul_(theta)
    term3 = s.outer(s).div_(sy)

    H -= term1
    H += term2
    H *= u
    H += term3
    return H


class SSVM(HessianUpdateStrategy):
    """This one is from Oren, S. S., & Spedicato, E. (1976). Optimal conditioning of self-scaling variable Metric algorithms. Mathematical Programming, 10(1), 70–90. doi:10.1007/bf01580654
    """
    def __init__(
        self,
        switch: tuple[float,float] | Literal[1,2,3,4] = 3,
        init_scale: float | Literal["auto"] = 'auto',
        tol: float = 1e-10,
        tol_reset: bool = True,
        reset_interval: int | None = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(switch=switch)
        super().__init__(
            defaults=defaults,
            init_scale=init_scale,
            tol=tol,
            tol_reset=tol_reset,
            reset_interval=reset_interval,
            beta=beta,
            update_freq=update_freq,
            scale_first=scale_first,
            scale_second=scale_second,
            concat_params=concat_params,
            inverse=True,
            inner=inner,
        )

    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return ssvm_H_(H=H, s=s, y=y, g=g, switch=settings['switch'], tol=settings['tol'])