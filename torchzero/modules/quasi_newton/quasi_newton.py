"""Use BFGS or maybe SR1."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal

import torch

from ...core import Chainable, Module, TensorwiseTransform, Transform
from ...utils import TensorList, set_storage_, unpack_states


def _safe_dict_update_(d1_:dict, d2:dict):
    inter = set(d1_.keys()).intersection(d2.keys())
    if len(inter) > 0: raise RuntimeError(f"Duplicate keys {inter}")
    d1_.update(d2)

def _maybe_lerp_(state, key, value: torch.Tensor, beta: float | None):
    if (beta is None) or (beta == 0) or (key not in state): state[key] = value
    elif state[key].shape != value.shape: state[key] = value
    else: state[key].lerp_(value, 1-beta)

class HessianUpdateStrategy(TensorwiseTransform, ABC):
    """Base class for quasi-newton methods that store and update hessian approximation H or inverse B.

    This is an abstract class, to use it, subclass it and override `update_H` and/or `update_B`.

    Args:
        defaults (dict | None, optional): defaults. Defaults to None.
        init_scale (float | Literal["auto"], optional):
            initial hessian matrix is set to identity times this.

            "auto" corresponds to a heuristic from Nocedal. Stephen J. Wright. Numerical Optimization p.142-143.

            Defaults to "auto".
        tol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional): whether to reset the hessian approximation when tolerance is not met. Defaults to False.
        reset_interval (int | None | Literal["auto"], optional):
            interval between resetting the hessian approximation.

            "auto" corresponds to number of decision variables + 1.

            None - no resets.

            Defaults to None.
        beta (float | None, optional): momentum on H or B. Defaults to None.
        update_freq (int, optional): frequency of updating H or B. Defaults to 1.
        scale_first (bool, optional):
            whether to downscale first step before hessian approximation becomes available. Defaults to True.
        scale_second (bool, optional): whether to downscale second step. Defaults to False.
        concat_params (bool, optional):
            If true, all parameters are treated as a single vector.
            If False, the update rule is applied to each parameter separately. Defaults to True.
        inverse (bool, optional):
            set to True if this method uses hessian inverse approximation H and has `update_H` method.
            set to False if this maintains hessian approximation B and has `update_B method`.
            Defaults to True.
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.

    Example:
        Implementing BFGS method that maintains an estimate of the hessian inverse (H):

        .. code-block:: python

            class BFGS(HessianUpdateStrategy):
                def __init__(
                    self,
                    init_scale: float | Literal["auto"] = "auto",
                    tol: float = 1e-10,
                    tol_reset: bool = False,
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

                def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
                    tol = settings["tol"]
                    sy = torch.dot(s, y)
                    if sy <= tol: return H
                    num1 = (sy + (y @ H @ y)) * s.outer(s)
                    term1 = num1.div_(sy**2)
                    num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
                    term2 = num2.div_(sy)
                    H += term1.sub_(term2)
                    return H

    """
    def __init__(
        self,
        defaults: dict | None = None,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        tol_reset: bool = False,
        reset_interval: int | None | Literal['auto'] = None,
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

    def _reset_M_(self, M: torch.Tensor, s:torch.Tensor,y:torch.Tensor, inverse:bool, init_scale: Any, state:dict[str,Any]):
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
    def update_tensor(self, tensor, param, grad, loss, state, settings):
        p = param.view(-1); g = tensor.view(-1)
        inverse = settings['inverse']
        M_key = 'H' if inverse else 'B'
        M = state.get(M_key, None)
        step = state.get('step', 0) + 1
        state['step'] = step
        init_scale = settings['init_scale']
        tol = settings['tol']
        tol_reset = settings['tol_reset']
        reset_interval = settings['reset_interval']
        if reset_interval == 'auto': reset_interval = tensor.numel() + 1

        if M is None:
            M = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1:
                if inverse: M /= init_scale
                else: M *= init_scale

            state[M_key] = M
            state['f_prev'] = loss
            state['p_prev'] = p.clone()
            state['g_prev'] = g.clone()
            return

        state['f'] = loss
        p_prev = state['p_prev']
        g_prev = state['g_prev']
        s: torch.Tensor = p - p_prev
        y: torch.Tensor = g - g_prev
        state['p_prev'].copy_(p)
        state['g_prev'].copy_(g)

        if reset_interval is not None and step % reset_interval == 0:
            self._reset_M_(M, s, y, inverse, init_scale, state)
            return

        # tolerance on gradient difference to avoid exploding after converging
        if y.abs().max() <= tol:
            # reset history
            if tol_reset: self._reset_M_(M, s, y, inverse, init_scale, state)
            return

        if step == 2 and init_scale == 'auto':
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

        state['f_prev'] = loss

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, loss, state, settings):
        step = state.get('step', 0)

        if settings['scale_second'] and step == 2:
            scale_factor = 1 / tensor.abs().sum().clip(min=1)
            scale_factor = scale_factor.clip(min=torch.finfo(tensor.dtype).eps)
            tensor = tensor * scale_factor

        inverse = settings['inverse']
        if inverse:
            H = state['H']
            return (H @ tensor.view(-1)).view_as(tensor)

        B = state['B']

        return torch.linalg.solve_ex(B, tensor.view(-1))[0].view_as(tensor) # pylint:disable=not-callable

    # def post_step(self, var):
    #     if self._concat_params:
    #         param = var.params[0]
    #         if self.settings[param]['inverse']:
    #             var.storage['hessian inverse'] = self.state[param]['H']
    #         else:
    #             var.storage['hessian'] = self.state[param]['B']

class _HessianUpdateStrategyDefaults(HessianUpdateStrategy):
    '''This is :code:`HessianUpdateStrategy` subclass for algorithms with no extra defaults, to skip the lengthy __init__.
    Refer to :code:`HessianUpdateStrategy` documentation.

    Example:
        Implementing BFGS method that maintains an estimate of the hessian inverse (H):

        .. code-block:: python

            class BFGS(_HessianUpdateStrategyDefaults):
                """Broyden–Fletcher–Goldfarb–Shanno algorithm"""
                def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
                    tol = settings["tol"]
                    sy = torch.dot(s, y)
                    if sy <= tol: return H
                    num1 = (sy + (y @ H @ y)) * s.outer(s)
                    term1 = num1.div_(sy**2)
                    num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
                    term2 = num2.div_(sy)
                    H += term1.sub_(term2)
                    return H

    Make sure to put at least a basic class level docstring to overwrite this.
    '''
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        tol_reset: bool = False,
        reset_interval: int | None = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inverse: bool = True,
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
            inverse=inverse,
            inner=inner,
        )
# ----------------------------------- BFGS ----------------------------------- #
def bfgs_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = torch.dot(s, y)
    if sy <= tol: return H
    num1 = (sy + (y @ H @ y)) * s.outer(s)
    term1 = num1.div_(sy**2)
    num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
    term2 = num2.div_(sy)
    H += term1.sub_(term2)
    return H

class BFGS(_HessianUpdateStrategyDefaults):
    """Broyden–Fletcher–Goldfarb–Shanno Quasi-Newton method. This is usually the most stable quasi-newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe()` is recommended, although this can be stable without a line search. Alternatively warmup :code:`tz.m.Warmup` can stabilize quasi-newton methods without line search.

    .. warning::
        this uses roughly O(N^2) memory.

    Args:
        init_scale (float | Literal["auto"], optional):
            initial hessian matrix is set to identity times this.

            "auto" corresponds to a heuristic from Nocedal. Stephen J. Wright. Numerical Optimization p.142-143.

            Defaults to "auto".
        tol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional): whether to reset the hessian approximation when tolerance is not met. Defaults to True.
        reset_interval (int | None | Literal["auto"], optional):
            interval between resetting the hessian approximation.

            "auto" corresponds to number of decision variables + 1.

            None - no resets.

            Defaults to None.
        beta (float | None, optional): momentum on H or B. Defaults to None.
        update_freq (int, optional): frequency of updating H or B. Defaults to 1.
        scale_first (bool, optional):
            whether to downscale first step before hessian approximation becomes available. Defaults to True.
        scale_second (bool, optional): whether to downscale second step. Defaults to False.
        concat_params (bool, optional):
            If true, all parameters are treated as a single vector.
            If False, the update rule is applied to each parameter separately. Defaults to True.
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.

    Examples:
        BFGS with strong-wolfe line search:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.BFGS(),
                tz.m.StrongWolfe()
            )

        BFGS preconditioning applied to momentum:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.BFGS(inner=tz.m.EMA(0.9)),
                tz.m.LR(1e-2)
            )
    """

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

class SR1(_HessianUpdateStrategyDefaults):
    """Symmetric Rank 1 Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        approximate Hessians generated by the SR1 method show faster progress towards the true Hessian than other methods, but it is more unstable.

    .. note::
        SR1 doesn't enforce the hessian estimate to be positive definite, therefore it can generate directions that are not descent directions.

    .. warning::
        this uses roughly O(N^2) memory.

    Args:
        init_scale (float | Literal["auto"], optional):
            initial hessian matrix is set to identity times this.

            "auto" corresponds to a heuristic from Nocedal. Stephen J. Wright. Numerical Optimization p.142-143.

            Defaults to "auto".
        tol (float | None, optional):
            tolerance for minimal gradient difference to avoid instability. Defaults to 1e-10.
        tol_reset (bool, optional): whether to reset the hessian approximation when tolerance is not met. Defaults to True.
        reset_interval (int | None | Literal["auto"], optional):
            interval between resetting the hessian approximation.

            "auto" corresponds to number of decision variables + 1.

            None - no resets.

            Defaults to None.
        beta (float | None, optional): momentum on H or B. Defaults to None.
        update_freq (int, optional): frequency of updating H or B. Defaults to 1.
        scale_first (bool, optional):
            whether to downscale first step before hessian approximation becomes available. Defaults to True.
        scale_second (bool, optional): whether to downscale second step. Defaults to False.
        concat_params (bool, optional):
            If true, all parameters are treated as a single vector.
            If False, the update rule is applied to each parameter separately. Defaults to True.
        inner (Chainable | None, optional): preconditioning is applied to the output of this module. Defaults to None.

    Examples:
        SR1 with strong-wolfe line search

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SR1(),
                tz.m.StrongWolfe()
            )

        BFGS preconditioning applied to momentum

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.SR1(inner=tz.m.EMA(0.9)),
                tz.m.LR(1e-2)
            )
    """

    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return sr1_H_(H=H, s=s, y=y, tol=settings['tol'])
    def update_B(self, B, s, y, p, g, p_prev, g_prev, state, settings):
        return sr1_H_(H=B, s=y, y=s, tol=settings['tol'])

# ------------------------------------ DFP ----------------------------------- #
def dfp_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = torch.dot(s, y)
    if sy.abs() <= tol: return H
    term1 = torch.outer(s, s).div_(sy)
    yHy = torch.dot(y, H @ y) #
    if yHy.abs() <= tol: return H
    num = H @ torch.outer(y, y) @ H
    term2 = num.div_(yHy)
    H += term1.sub_(term2)
    return H

class DFP(_HessianUpdateStrategyDefaults):
    """Davidon–Fletcher–Powell Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return dfp_H_(H=H, s=s, y=y, tol=settings['tol'])


# formulas for methods below from Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
# H' = H - (Hy - S)c^T / c^T*y
# the difference is how `c` is calculated

def broyden_good_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    c = H.T @ s
    cy = c.dot(y)
    if cy.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/cy
    return H

def broyden_bad_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    c = y
    cy = c.dot(y)
    if cy.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/cy
    return H

def greenstadt1_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, g_prev: torch.Tensor, tol: float):
    c = g_prev
    cy = c.dot(y)
    if cy.abs() <= tol: return H
    num = (H@y).sub_(s).outer(c)
    H -= num/cy
    return H

def greenstadt2_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    Hy = H @ y
    c = H @ Hy # pylint:disable=not-callable
    cy = c.dot(y)
    if cy.abs() <= tol: return H
    num = Hy.sub_(s).outer(c)
    H -= num/cy
    return H

class BroydenGood(_HessianUpdateStrategyDefaults):
    """Broyden's "good" Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_good_H_(H=H, s=s, y=y, tol=settings['tol'])

class BroydenBad(_HessianUpdateStrategyDefaults):
    """Broyden's "bad" Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return broyden_bad_H_(H=H, s=s, y=y, tol=settings['tol'])

class Greenstadt1(_HessianUpdateStrategyDefaults):
    """Greenstadt's first Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt1_H_(H=H, s=s, y=y, g_prev=g_prev, tol=settings['tol'])

class Greenstadt2(_HessianUpdateStrategyDefaults):
    """Greenstadt's second Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472

    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return greenstadt2_H_(H=H, s=s, y=y, tol=settings['tol'])


def column_updating_H_(H:torch.Tensor, s:torch.Tensor, y:torch.Tensor, tol:float):
    j = y.abs().argmax()

    denom = y[j]
    if denom.abs() < tol: return H

    Hy = H @ y.unsqueeze(1)
    num = s.unsqueeze(1) - Hy

    H[:, j] += num.squeeze() / denom
    return H

class ColumnUpdatingMethod(_HessianUpdateStrategyDefaults):
    """
    Column-updating Quasi-Newton method. This is computationally cheaper than other Quasi-Newton methods
    due to only updating one column of the inverse hessian approximation per step.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Lopes, V. L., & Martínez, J. M. (1995). Convergence properties of the inverse column-updating method. Optimization Methods & Software, 6(2), 127–144. from https://www.ime.unicamp.br/sites/default/files/pesquisa/relatorios/rp-1993-76.pdf
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return column_updating_H_(H=H, s=s, y=y, tol=settings['tol'])

def thomas_H_(H: torch.Tensor, R:torch.Tensor, s: torch.Tensor, y: torch.Tensor, tol:float):
    s_norm = torch.linalg.vector_norm(s) # pylint:disable=not-callable
    I = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
    d = (R + I * (s_norm/2)) @ s
    ds = d.dot(s)
    if ds.abs() <= tol: return H, R
    R = (1 + s_norm) * ((I*s_norm).add_(R).sub_(d.outer(d).div_(ds)))

    c = H.T @ d
    cy = c.dot(y)
    if cy.abs() <= tol: return H, R
    num = (H@y).sub_(s).outer(c)
    H -= num/cy
    return H, R

class ThomasOptimalMethod(_HessianUpdateStrategyDefaults):
    """
    Thomas's "optimal" Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Thomas, Stephen Walter. Sequential estimation techniques for quasi-Newton algorithms. Cornell University, 1975.
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        if 'R' not in state: state['R'] = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
        H, state['R'] = thomas_H_(H=H, R=state['R'], s=s, y=y, tol=settings['tol'])
        return H

    def _reset_M_(self, M, s, y,inverse, init_scale, state):
        super()._reset_M_(M, s, y, inverse, init_scale, state)
        for st in self.state.values():
            st.pop("R", None)

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

# I couldn't find formula for H
class PSB(HessianUpdateStrategy):
    """Powell's Symmetric Broyden Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Spedicato, E., & Huang, Z. (1997). Numerical experience with newton-like methods for nonlinear algebraic systems. Computing, 58(1), 69–89. doi:10.1007/bf02684472
    """
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 'auto',
        tol: float = 1e-10,
        tol_reset: bool = False,
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


# Algorithms from Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171
def pearson_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    Hy = H@y
    yHy = y.dot(Hy)
    if yHy.abs() <= tol: return H
    num = (s - Hy).outer(Hy)
    H += num.div_(yHy)
    return H

class Pearson(_HessianUpdateStrategyDefaults):
    """
    Pearson's Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171.
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return pearson_H_(H=H, s=s, y=y, tol=settings['tol'])

def mccormick_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy.abs() <= tol: return H
    num = (s - H@y).outer(s)
    H += num.div_(sy)
    return H

class McCormick(_HessianUpdateStrategyDefaults):
    """McCormicks's Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171.

        This is "Algorithm 2", attributed to McCormick in this paper. However for some reason this method is also called Pearson's 2nd method in other sources.
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return mccormick_H_(H=H, s=s, y=y, tol=settings['tol'])

def projected_newton_raphson_H_(H: torch.Tensor, R:torch.Tensor, s: torch.Tensor, y: torch.Tensor, tol:float):
    Hy = H @ y
    yHy = y.dot(Hy)
    if yHy.abs() < tol: return H, R
    H -= Hy.outer(Hy) / yHy
    R += (s - R@y).outer(Hy) / yHy
    return H, R

class ProjectedNewtonRaphson(HessianUpdateStrategy):
    """
    Projected Newton Raphson method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        this is an experimental method.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Pearson, J. D. (1969). Variable metric methods of minimisation. The Computer Journal, 12(2), 171–178. doi:10.1093/comjnl/12.2.171.

        This one is Algorithm 7.
    """
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 'auto',
        tol: float = 1e-10,
        tol_reset: bool = False,
        reset_interval: int | None | Literal['auto'] = 'auto',
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
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
        if 'R' not in state: state['R'] = torch.eye(H.size(-1), device=H.device, dtype=H.dtype)
        H, R = projected_newton_raphson_H_(H=H, R=state['R'], s=s, y=y, tol=settings['tol'])
        state["R"] = R
        return H

    def _reset_M_(self, M, s, y, inverse, init_scale, state):
        assert inverse
        M.copy_(state["R"])

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
    """
    Self-scaling variable metric Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Oren, S. S., & Spedicato, E. (1976). Optimal conditioning of self-scaling variable Metric algorithms. Mathematical Programming, 10(1), 70–90. doi:10.1007/bf01580654
    """
    def __init__(
        self,
        switch: tuple[float,float] | Literal[1,2,3,4] = 3,
        init_scale: float | Literal["auto"] = 'auto',
        tol: float = 1e-10,
        tol_reset: bool = False,
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

# HOSHINO, S. (1972). A Formulation of Variable Metric Methods. IMA Journal of Applied Mathematics, 10(3), 394–403. doi:10.1093/imamat/10.3.394
def hoshino_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    Hy = H@y
    ys = y.dot(s)
    if ys.abs() <= tol: return H
    yHy = y.dot(Hy)
    denom = ys + yHy
    if denom.abs() <= tol: return H

    term1 = 1/denom
    term2 = s.outer(s).mul_(1 + ((2 * yHy) / ys))
    term3 = s.outer(y) @ H
    term4 = Hy.outer(s)
    term5 = Hy.outer(y) @ H

    inner_term = term2 - term3 - term4 - term5
    H += inner_term.mul_(term1)
    return H

def gradient_correction(g: TensorList, s: TensorList, y: TensorList):
    sy = s.dot(y)
    if sy.abs() < torch.finfo(g[0].dtype).eps: return g
    return g - (y * (s.dot(g) / sy))


class GradientCorrection(Transform):
    """
    Estimates gradient at minima along search direction assuming function is quadratic.

    This can useful as inner module for second order methods with inexact line search.

    Example:
        L-BFGS with gradient correction

        .. code-block :: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.LBFGS(inner=tz.m.GradientCorrection()),
                tz.m.Backtracking()
            )

    Reference:
        HOSHINO, S. (1972). A Formulation of Variable Metric Methods. IMA Journal of Applied Mathematics, 10(3), 394–403. doi:10.1093/imamat/10.3.394

    """
    def __init__(self):
        super().__init__(None, uses_grad=False)

    def apply(self, tensors, params, grads, loss, states, settings):
        if 'p_prev' not in states[0]:
            p_prev = unpack_states(states, tensors, 'p_prev', init=params)
            g_prev = unpack_states(states, tensors, 'g_prev', init=tensors)
            return tensors

        p_prev, g_prev = unpack_states(states, tensors, 'p_prev', 'g_prev', cls=TensorList)
        g_hat = gradient_correction(TensorList(tensors), params-p_prev, tensors-g_prev)

        p_prev.copy_(params)
        g_prev.copy_(tensors)
        return g_hat

class Horisho(_HessianUpdateStrategyDefaults):
    """
    Horisho's variable metric Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        HOSHINO, S. (1972). A Formulation of Variable Metric Methods. IMA Journal of Applied Mathematics, 10(3), 394–403. doi:10.1093/imamat/10.3.394
    """

    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return hoshino_H_(H=H, s=s, y=y, tol=settings['tol'])

# Fletcher, R. (1970). A new approach to variable metric algorithms. The Computer Journal, 13(3), 317–322. doi:10.1093/comjnl/13.3.317
def fletcher_vmm_H_(H:torch.Tensor, s: torch.Tensor, y:torch.Tensor, tol: float):
    sy = s.dot(y)
    if sy.abs() < tol: return H
    Hy = H @ y

    term1 = (s.outer(y) @ H).div_(sy)
    term2 = (Hy.outer(s)).div_(sy)
    term3 = 1 + (y.dot(Hy) / sy)
    term4 = s.outer(s).div_(sy)

    H -= (term1 + term2 - term4.mul_(term3))
    return H

class FletcherVMM(_HessianUpdateStrategyDefaults):
    """
    Fletcher's variable metric Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is highly recommended.

    .. note::
        BFGS is the recommended QN method and will usually outperform this.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Fletcher, R. (1970). A new approach to variable metric algorithms. The Computer Journal, 13(3), 317–322. doi:10.1093/comjnl/13.3.317
    """
    def update_H(self, H, s, y, p, g, p_prev, g_prev, state, settings):
        return fletcher_vmm_H_(H=H, s=s, y=y, tol=settings['tol'])


# Moghrabi, I. A., Hassan, B. A., & Askar, A. (2022). New self-scaling quasi-newton methods for unconstrained optimization. Int. J. Math. Comput. Sci., 17, 1061U.
def new_ssm1(H: torch.Tensor, s: torch.Tensor, y: torch.Tensor, f, f_prev, tol: float, type:int):
    sy = s.dot(y)
    if sy < tol: return H

    term1 = (H @ y.outer(s) + s.outer(y) @ H) / sy

    if type == 1:
        pba = (2*sy + 2*(f-f_prev)) / sy

    elif type == 2:
        pba = (f_prev - f + 1/(2*sy)) / sy

    else:
        raise RuntimeError(type)

    term3 = 1/pba + y.dot(H@y) / sy
    term4 = s.outer(s) / sy

    H.sub_(term1)
    H.add_(term4.mul_(term3))
    return H


class NewSSM(HessianUpdateStrategy):
    """Self-scaling Quasi-Newton method.

    .. note::
        a line search such as :code:`tz.m.StrongWolfe(plus_minus=True)` is required.

    .. warning::
        this uses roughly O(N^2) memory.

    Reference:
        Moghrabi, I. A., Hassan, B. A., & Askar, A. (2022). New self-scaling quasi-newton methods for unconstrained optimization. Int. J. Math. Comput. Sci., 17, 1061U.
    """
    def __init__(
        self,
        type: Literal[1, 2] = 1,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        tol_reset: bool = False,
        reset_interval: int | None = None,
        beta: float | None = None,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = False,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
            defaults=dict(type=type),
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
        f = state['f']
        f_prev = state['f_prev']
        return new_ssm1(H=H, s=s, y=y, f=f, f_prev=f_prev, type=settings['type'], tol=settings['tol'])

