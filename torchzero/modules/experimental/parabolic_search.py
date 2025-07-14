import math
from collections.abc import Mapping
from operator import itemgetter

import torch

from ...core import Module
from ...utils import TensorList



def adaptive_tracking(
    f,
    f_0,
    f_1,
    t_0,
    maxiter: int
):

    t = t_0
    f_t = f(t)

    # backtrack
    if f_t > f_0:
        if f_1 > f_0: t = min(0.5, t_0/2)
        while f_t > f_0:
            maxiter -= 1
            if maxiter < 0: return 0, f_0
            t = t/2
            f_t = f(t) if t!=1 else f_1
        return t, f_t

    # forwardtrack
    f_prev = f_t
    t *= 2
    f_t = f(t)
    if f_prev < f_t: return t/2, f_prev
    while f_prev >= f_t:
        maxiter -= 1
        if maxiter < 0: return t, f_t
        f_prev = f_t
        t *= 2
        f_t = f(t)
    return t/2, f_prev



class ParabolaSearch(Module):
    """"""
    def __init__(
        self,
        step_size: float = 1e-2,
        adaptive: bool=True,
        normalize: bool=False,
        # method: str | None = None,
        maxiter: int | None = 10,
        # bracket=None,
        # bounds=None,
        # tol: float | None = None,
        # options=None,
    ):
        if normalize and adaptive: raise ValueError("pick either normalize or adaptive")
        defaults = dict(step_size=step_size, adaptive=adaptive, normalize=normalize, maxiter=maxiter)
        super().__init__(defaults)

        import scipy.optimize
        self.scopt = scipy.optimize


    @torch.no_grad
    def step(self, var):
        x_0 = TensorList(var.params)
        closure = var.closure
        assert closure is not None
        settings = self.settings[x_0[0]]
        step_size = settings['step_size']
        adaptive = settings['adaptive']
        normalize = settings['normalize']
        maxiter = settings['maxiter']
        if normalize and adaptive: raise ValueError("pick either normalize or adaptive")

        grad = TensorList(var.get_grad())
        f_0 = var.get_loss(False)

        scale = 1
        if normalize: grad = grad/grad.abs().mean().clip(min=1e-8)
        if adaptive: scale = grad.abs().mean().clip(min=1e-8)

        # make step
        v_0 = grad * (step_size/scale)
        x_0 -= v_0
        with torch.enable_grad():
            f_1 = closure()
            grad = x_0.grad

        x_0 += v_0
        if normalize: grad = grad/grad.abs().mean().clip(min=1e-8)
        v_1 = grad * (step_size/scale)
        a = v_1 - v_0

        def parabolic_objective(t: float):
            nonlocal x_0

            step = v_0*t + 0.5*a*t**2
            x_0 -= step
            value = closure(False)
            x_0 += step
            return value.detach().cpu()

        prev_t = self.global_state.get('prev_t', 2)
        t, f = adaptive_tracking(parabolic_objective, f_0=f_0, f_1=f_1, t_0=prev_t, maxiter=maxiter)
        self.global_state['prev_t'] = t

        # method, bracket, bounds, tol, options, maxiter = itemgetter(
        #     'method', 'bracket', 'bounds', 'tol', 'options', 'maxiter')(self.settings[var.params[0]])

        # if maxiter is not None:
        #     options = dict(options) if isinstance(options, Mapping) else {}
        #     options['maxiter'] = maxiter

        # res = self.scopt.minimize_scalar(parabolic_objective, method=method, bracket=bracket, bounds=bounds, tol=tol, options=options)
        # t = res.x

        var.update = v_0*t + 0.5*a*t**2
        return var

class CubicParabolaSearch(Module):
    """"""
    def __init__(
        self,
        step_size: float = 1e-2,
        adaptive: bool=True,
        normalize: bool=False,
        # method: str | None = None,
        maxiter: int | None = 10,
        # bracket=None,
        # bounds=None,
        # tol: float | None = None,
        # options=None,
    ):
        if normalize and adaptive: raise ValueError("pick either normalize or adaptive")
        defaults = dict(step_size=step_size, adaptive=adaptive, normalize=normalize, maxiter=maxiter)
        super().__init__(defaults)

        import scipy.optimize
        self.scopt = scipy.optimize


    @torch.no_grad
    def step(self, var):
        x_0 = TensorList(var.params)
        closure = var.closure
        assert closure is not None
        settings = self.settings[x_0[0]]
        step_size = settings['step_size']
        adaptive = settings['adaptive']
        maxiter = settings['maxiter']
        normalize = settings['normalize']
        if normalize and adaptive: raise ValueError("pick either normalize or adaptive")

        grad = TensorList(var.get_grad())
        f_0 = var.get_loss(False)

        scale = 1
        if normalize: grad = grad/grad.abs().mean().clip(min=1e-8)
        if adaptive: scale = grad.abs().mean().clip(min=1e-8)

        # make step
        v_0 = grad * (step_size/scale)
        x_0 -= v_0
        with torch.enable_grad():
            f_1 = closure()
            grad = x_0.grad

        if normalize: grad = grad/grad.abs().mean().clip(min=1e-8)
        v_1 = grad * (step_size/scale)
        a_0 = v_1 - v_0

        # make another step
        x_0 -= v_1
        with torch.enable_grad():
            f_2 = closure()
            grad = x_0.grad

        if normalize: grad = grad/grad.abs().mean().clip(min=1e-8)
        v_2 = grad * (step_size/scale)
        a_1 = v_2 - v_1

        j = a_1 - a_0

        x_0 += v_0
        x_0 += v_1

        def parabolic_objective(t: float):
            nonlocal x_0

            step = v_0*t + (1/2)*a_0*t**2 + (1/6)*j*t**3
            x_0 -= step
            value = closure(False)
            x_0 += step
            return value


        prev_t = self.global_state.get('prev_t', 2)
        t, f = adaptive_tracking(parabolic_objective, f_0=f_0, f_1=f_1, t_0=prev_t, maxiter=maxiter)
        self.global_state['prev_t'] = t

        # method, bracket, bounds, tol, options, maxiter = itemgetter(
        #     'method', 'bracket', 'bounds', 'tol', 'options', 'maxiter')(self.settings[var.params[0]])

        # if maxiter is not None:
        #     options = dict(options) if isinstance(options, Mapping) else {}
        #     options['maxiter'] = maxiter

        # res = self.scopt.minimize_scalar(parabolic_objective, method=method, bracket=bracket, bounds=bounds, tol=tol, options=options)
        # t = res.x

        var.update = v_0*t + (1/2)*a_0*t**2 + (1/6)*j*t**3
        return var

