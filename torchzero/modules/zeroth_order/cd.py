import math
import random
import warnings
from functools import partial
from typing import Literal

import numpy as np
import torch

from ...core import Module
from ...utils import NumberList, TensorList
from ..line_search.adaptive import adaptive_tracking

class CD(Module):
    """Coordinate descent. Proposes a descent direction along a single coordinate.
    You can then put a line search such as ``tz.m.ScipyMinimizeScalar``, or a fixed step size.

    Args:
        h (float, optional): finite difference step size. Defaults to 1e-3.
        grad (bool, optional):
            if True, scales direction by gradient estimate. If False, the scale is fixed to 1. Defaults to True.
        adaptive (bool, optional):
            whether to adapt finite difference step size, this requires an additional buffer. Defaults to True.
        index (str, optional):
            index selection strategy.
            - "cyclic" - repeatedly cycles through each coordinate, e.g. ``1,2,3,1,2,3,...``.
            - "cyclic2" - cycles forward and then backward, e.g ``1,2,3,3,2,1,1,2,3,...`` (default).
            - "random" - picks coordinate randomly.
        threepoint (bool, optional):
            whether to use three points (three function evaluatins) to determine descent direction.
            if False, uses two points, but then ``adaptive`` can't be used. Defaults to True.
    """
    def __init__(self, h:float=1e-3, grad:bool=True, adaptive:bool=True, index:Literal['cyclic', 'cyclic2', 'random']="cyclic2", threepoint:bool=True,):
        defaults = dict(h=h, grad=grad, adaptive=adaptive, index=index, threepoint=threepoint)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None:
            raise RuntimeError("CD requires closure")

        params = TensorList(var.params)
        ndim = params.global_numel()

        grad_step_size = self.defaults['grad']
        adaptive = self.defaults['adaptive']
        index_strategy = self.defaults['index']
        h = self.defaults['h']
        threepoint = self.defaults['threepoint']

        # ------------------------------ determine index ----------------------------- #
        if index_strategy == 'cyclic':
            idx = self.global_state.get('idx', 0) % ndim
            self.global_state['idx'] = idx + 1

        elif index_strategy == 'cyclic2':
            idx = self.global_state.get('idx', 0)
            self.global_state['idx'] = idx + 1
            if idx >= ndim * 2:
                idx = self.global_state['idx'] = 0
            if idx >= ndim:
                idx  = (2*ndim - idx) - 1

        elif index_strategy == 'random':
            if 'generator' not in self.global_state:
                self.global_state['generator'] = random.Random(0)
            generator = self.global_state['generator']
            idx = generator.randrange(0, ndim)

        else:
            raise ValueError(index_strategy)

        # -------------------------- find descent direction -------------------------- #
        h_vec = None
        if adaptive:
            if threepoint:
                h_vec = self.get_state(params, 'h_vec', init=lambda x: torch.full_like(x, h), cls=TensorList)
                h = float(h_vec.flat_get(idx))
            else:
                warnings.warn("CD adaptive=True only works with threepoint=True")

        f_0 = var.get_loss(False)
        params.flat_set_lambda_(idx, lambda x: x + h)
        f_p = closure(False)

        # -------------------------------- threepoint -------------------------------- #
        if threepoint:
            params.flat_set_lambda_(idx, lambda x: x - 2*h)
            f_n = closure(False)
            params.flat_set_lambda_(idx, lambda x: x + h)

            if adaptive:
                assert h_vec is not None
                if f_0 <= f_p and f_0 <= f_n:
                    h_vec.flat_set_lambda_(idx, lambda x: max(x/2, 1e-10))
                else:
                    if abs(f_0 - f_n) < 1e-12 or abs((f_p - f_0) / (f_0 - f_n) - 1) < 1e-2:
                        h_vec.flat_set_lambda_(idx, lambda x: min(x*2, 1e10))

            if grad_step_size:
                alpha = (f_p - f_n) / (2*h)

            else:
                if f_0 < f_p and f_0 < f_n: alpha = 0
                elif f_p < f_n: alpha = -1
                else: alpha = 1

        # --------------------------------- twopoint --------------------------------- #
        else:
            params.flat_set_lambda_(idx, lambda x: x - h)
            if grad_step_size:
                alpha = (f_p - f_0) / h
            else:
                if f_p < f_0: alpha = -1
                else: alpha = 1

        # ----------------------------- create the update ---------------------------- #
        update = params.zeros_like()
        update.flat_set_(idx, alpha)
        var.update = update
        return var


def _icd_get_idx(self: Module, params: TensorList):
    ndim = params.global_numel()
    igrad = self.get_state(params, "igrad", cls=TensorList)

    # -------------------------- 1st n steps fill igrad -------------------------- #
    index = self.global_state.get('index', 0)
    self.global_state['index'] = index + 1
    if index < ndim:
        return index, igrad

    # ------------------ select randomly weighted by magnitudes ------------------ #
    igrad_abs = igrad.abs()
    gmin = igrad_abs.global_min()
    gmax = igrad_abs.global_max()

    pmin, pmax, pow = self.get_settings(params, "pmin", "pmax", "pow", cls=NumberList)

    p: TensorList = ((igrad_abs - gmin) / (gmax - gmin)) ** pow # pyright:ignore[reportOperatorIssue]
    p.mul_(pmax-pmin).add_(pmin)

    if 'np_gen' not in self.global_state:
        self.global_state['np_gen'] = np.random.default_rng(0)
    np_gen = self.global_state['np_gen']

    p_vec = p.to_vec()
    p_sum = p_vec.sum()
    if p_sum > 1e-12:
        return np_gen.choice(ndim, p=p_vec.div_(p_sum).numpy(force=True)), igrad

    # --------------------- sum is too small, do cycle again --------------------- #
    self.global_state.clear()
    self.clear_state_keys('h_vec', 'igrad', 'alphas')

    if 'generator' not in self.global_state:
        self.global_state['generator'] = random.Random(0)
    generator = self.global_state['generator']
    return generator.randrange(0, p_vec.numel()), igrad

class CCD(Module):
    """Cumulative coordinate descent. This updates one gradient coordinate at a time and accumulates it
    to the update direction. The coordinate updated is random weighted by magnitudes of current update direction.
    As update direction ceases to be a descent direction due to old accumulated coordinates, it is decayed.

    Args:
        pmin (float, optional): multiplier to probability of picking the lowest magnitude gradient. Defaults to 0.1.
        pmax (float, optional): multiplier to probability of picking the largest magnitude gradient. Defaults to 1.0.
        pow (int, optional): power transform to probabilities. Defaults to 2.
        decay (float, optional): accumulated gradient decay on failed step. Defaults to 0.5.
        decay2 (float, optional): decay multiplier decay on failed step. Defaults to 0.25.
        nplus (float, optional): step size increase on successful steps. Defaults to 1.5.
        nminus (float, optional): step size increase on unsuccessful steps. Defaults to 0.75.
    """
    def __init__(self, pmin=0.1, pmax=1.0, pow=2, decay:float=0.8, decay2:float=0.2, nplus=1.5, nminus=0.75):

        defaults = dict(pmin=pmin, pmax=pmax, pow=pow, decay=decay, decay2=decay2, nplus=nplus, nminus=nminus)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None:
            raise RuntimeError("CD requires closure")

        params = TensorList(var.params)
        p_prev = self.get_state(params, "p_prev", init=params, cls=TensorList)

        f_0 = var.get_loss(False)
        step_size = self.global_state.get('step_size', 1)

        # ------------------------ hard reset on infinite loss ----------------------- #
        if not math.isfinite(f_0):
            del self.global_state['f_prev']
            var.update = params - p_prev
            self.global_state.clear()
            self.state.clear()
            self.global_state["step_size"] = step_size / 10
            return var

        # ---------------------------- soft reset if stuck --------------------------- #
        if "igrad" in self.state[params[0]]:
            n_bad = self.global_state.get('n_bad', 0)

            f_prev = self.global_state.get("f_prev", None)
            if f_prev is not None:

                decay2 = self.defaults["decay2"]
                decay = self.global_state.get("decay", self.defaults["decay"])

                if f_0 >= f_prev:

                    igrad = self.get_state(params, "igrad", cls=TensorList)
                    del self.global_state['f_prev']

                    # undo previous update
                    var.update = params - p_prev

                    # increment n_bad
                    self.global_state['n_bad'] = n_bad + 1

                    # decay step size
                    self.global_state['step_size'] = step_size * self.defaults["nminus"]

                    # soft reset
                    if n_bad > 0:
                        igrad *= decay
                        self.global_state["decay"] = decay*decay2
                        self.global_state['n_bad'] = 0

                    return var

                else:
                    # increase step size and reset n_bad
                    self.global_state['step_size'] = step_size * self.defaults["nplus"]
                    self.global_state['n_bad'] = 0
                    self.global_state["decay"] = self.defaults["decay"]

            self.global_state['f_prev'] = float(f_0)

        # ------------------------------ determine index ----------------------------- #
        idx, igrad = _icd_get_idx(self, params)

        # -------------------------- find descent direction -------------------------- #
        h_vec = self.get_state(params, 'h_vec', init=lambda x: torch.full_like(x, 1e-3), cls=TensorList)
        h = float(h_vec.flat_get(idx))

        params.flat_set_lambda_(idx, lambda x: x + h)
        f_p = closure(False)

        params.flat_set_lambda_(idx, lambda x: x - 2*h)
        f_n = closure(False)
        params.flat_set_lambda_(idx, lambda x: x + h)

        # ---------------------------------- adapt h --------------------------------- #
        if f_0 <= f_p and f_0 <= f_n:
            h_vec.flat_set_lambda_(idx, lambda x: max(x/2, 1e-10))
        else:
            if abs(f_0 - f_n) < 1e-12 or abs((f_p - f_0) / (f_0 - f_n) - 1) < 1e-2:
                h_vec.flat_set_lambda_(idx, lambda x: min(x*2, 1e10))

        # ------------------------------- update igrad ------------------------------- #
        if f_0 < f_p and f_0 < f_n: alpha = 0
        else: alpha = (f_p - f_n) / (2*h)

        igrad.flat_set_(idx, alpha)

        # ----------------------------- create the update ---------------------------- #
        var.update = igrad * step_size
        p_prev.copy_(params)
        return var


class CCDLS(Module):
    """CCD with line search instead of adaptive step size.

    Args:
        pmin (float, optional): multiplier to probability of picking the lowest magnitude gradient. Defaults to 0.1.
        pmax (float, optional): multiplier to probability of picking the largest magnitude gradient. Defaults to 1.0.
        pow (int, optional): power transform to probabilities. Defaults to 2.
        decay (float, optional): accumulated gradient decay on failed step. Defaults to 0.5.
        decay2 (float, optional): decay multiplier decay on failed step. Defaults to 0.25.
        maxiter (int, optional): max number of line search iterations.
    """
    def __init__(self, pmin=0.1, pmax=1.0, pow=2, decay=0.8, decay2=0.2, maxiter=10, ):
        defaults = dict(pmin=pmin, pmax=pmax, pow=pow, maxiter=maxiter, decay=decay, decay2=decay2)
        super().__init__(defaults)

    @torch.no_grad
    def step(self, var):
        closure = var.closure
        if closure is None:
            raise RuntimeError("CD requires closure")

        params = TensorList(var.params)
        finfo = torch.finfo(params[0].dtype)
        f_0 = var.get_loss(False)

        # ------------------------------ determine index ----------------------------- #
        idx, igrad = _icd_get_idx(self, params)

        # -------------------------- find descent direction -------------------------- #
        h_vec = self.get_state(params, 'h_vec', init=lambda x: torch.full_like(x, 1e-3), cls=TensorList)
        h = float(h_vec.flat_get(idx))

        params.flat_set_lambda_(idx, lambda x: x + h)
        f_p = closure(False)

        params.flat_set_lambda_(idx, lambda x: x - 2*h)
        f_n = closure(False)
        params.flat_set_lambda_(idx, lambda x: x + h)

        # ---------------------------------- adapt h --------------------------------- #
        if f_0 <= f_p and f_0 <= f_n:
            h_vec.flat_set_lambda_(idx, lambda x: max(x/2, finfo.tiny * 2))
        else:
            # here eps, not tiny
            if abs(f_0 - f_n) < finfo.eps or abs((f_p - f_0) / (f_0 - f_n) - 1) < 1e-2:
                h_vec.flat_set_lambda_(idx, lambda x: min(x*2, finfo.max / 2))

        # ------------------------------- update igrad ------------------------------- #
        if f_0 < f_p and f_0 < f_n: alpha = 0
        else: alpha = (f_p - f_n) / (2*h)

        igrad.flat_set_(idx, alpha)

        # -------------------------------- line search ------------------------------- #
        x0 = params.clone()
        def f(a):
            params.sub_(igrad, alpha=a)
            loss = closure(False)
            params.copy_(x0)
            return loss

        a_prev = self.global_state.get('a_prev', 1)
        a, f_a, niter = adaptive_tracking(f, a_prev, maxiter=self.defaults['maxiter'], f_0=f_0)
        if (a is None) or (not math.isfinite(a)) or (not math.isfinite(f_a)):
            a = 0

        # -------------------------------- set a_prev -------------------------------- #
        decay2 = self.defaults["decay2"]
        decay = self.global_state.get("decay", self.defaults["decay"])

        if abs(a) > finfo.tiny * 2:
            assert f_a < f_0
            self.global_state['a_prev'] = max(min(a, finfo.max / 2), finfo.tiny * 2)
            self.global_state["decay"] = self.defaults["decay"]

        # ---------------------------- soft reset on fail ---------------------------- #
        else:
            igrad *= decay
            self.global_state["decay"] = decay*decay2
            self.global_state['a_prev'] = a_prev / 2

        # -------------------------------- set update -------------------------------- #
        var.update = igrad * a
        return var