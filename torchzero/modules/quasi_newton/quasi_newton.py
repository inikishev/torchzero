from typing import Any, Literal

import torch

from ...core import Chainable, Module, Preconditioner, TensorwisePreconditioner
from ...utils import TensorList


class BFGS(TensorwisePreconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        update_freq: int = 1,
        scale_first: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init_scale=init_scale, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: H *= init_scale
            state['H'] = H
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
            if ys != 0 and yy != 0: H *= ys/yy

        # BFGS update
        sy = torch.dot(s, y)
        if sy > 1e-10:
            num1 = (sy + (y @ H @ y)) * (torch.outer(s, s))
            term1 = num1.div_(sy**2)
            num2 = (torch.outer(H @ y, s).add_(torch.outer(s, y) @ H))
            term2 = num2.div_(sy)
            H += term1.sub_(term2)

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        state['step'] = state.get('step', 0) + 1
        H = state['H']
        return H @ tensor


class SR1(TensorwisePreconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 1,
        tol: float = 1e-10,
        eps: float = 1e-8,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(eps=eps, init_scale=init_scale, tol=tol, scale_second=scale_second)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']
        eps = settings['tol']

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: H *= init_scale
            state['H'] = H
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
            if ys != 0 and yy != 0: H *= ys/yy

        # SR1 update
        z = s - H@y
        denom = torch.dot(z, y)

        # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
        if denom.abs() >= eps * torch.linalg.norm(y) * torch.linalg.norm(z): # pylint:disable=not-callable
            H += torch.outer(z, z).div_(denom)

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        """precondition"""
        step = state['step'] = state.get('step', 0) + 1
        H = state['H']
        if settings['scale_second'] and step == 2:
            tensor = tensor/max(1, tensor.abs().sum()) # pyright:ignore[reportArgumentType]
        return H @ tensor



class DiagonalBFGS(Preconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        res_beta: float | None = None,
        H_beta: float | None = None,
        growth_clip: float | None = None,
        scale_first: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init_scale=init_scale, res_beta=res_beta, H_beta=H_beta, growth_clip=growth_clip, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=False, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update(self, tensors, params, grads, states, settings):
        p = TensorList(params); g = TensorList(tensors)
        step = self.global_state.get('step', 0)

        if any('H' not in s for s in states):
            for param, grad, state, setting in zip(params, tensors, states, settings):
                init_scale = setting['init_scale']
                if 'H' not in state:
                    state['H'] = torch.ones_like(param)
                if isinstance(init_scale, (int, float)) and init_scale != 1: state['H'] *= init_scale
                state['p_prev'] = param.clone()
                state['g_prev'] = grad.clone()
            return

        H = TensorList(s['H'] for s in states)
        p_prev = TensorList(s['p_prev'] for s in states)
        g_prev = TensorList(s['g_prev'] for s in states)
        s = p - p_prev
        y = g - g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y.abs().global_max() <= settings[0]['tol']:
            return

        if step == 1 and settings[0]['init_scale'] == 'auto':
            ys = y.dot(s)
            yy = y.dot(y)
            if ys != 0 and yy != 0: H *= ys/yy


        sy = s.dot(y)

        # num1 = (skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))
        # denom1 = skyk**2
        # term1 = num1 / denom1
        # num2 = (torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H)
        # term2 = num2 / skyk
        # H += term1 - term2

        if sy > 1e-10:
            num1 = (sy + (y * H).dot(y)) * s * s
            denom1 = sy**2
            term1 = num1 / denom1
            z = H * y * s
            num2 = z + z
            term2 = num2 / sy
            res = term1 - term2

            if settings[0]['res_beta'] is not None:
                for s,param in zip(states, params):
                    if 'res' not in s:
                        s['res'] = torch.zeros_like(param)
                res = TensorList(s['res'] for s in states)
                res.lerp_(res, 1-settings[0]['res_beta'])

            if settings[0]['growth_clip'] is not None:
                norm = res.global_vector_norm()

                if 'prev_norm' in self.global_state:
                    prev_norm = self.global_state['prev_norm']
                    allowed_norm = prev_norm * settings[0]['res_beta']
                    if norm > allowed_norm:
                        mul = (allowed_norm/norm).clip_(min=1e-5)
                        res.mul_(mul)
                        norm = norm * mul

                self.global_state['prev_norm'] = norm

            if settings[0]['H_beta'] is None: H += res
            else: H.lerp_(H + res, weight=1-settings[0]['H_beta'])

        p_prev.copy_(p)
        g_prev.copy_(g)

    @torch.no_grad
    def apply(self, tensors, params, grads, states, settings):
        self.global_state['step'] = self.global_state.get('step', 0) + 1
        return TensorList(tensors).mul_([s['H'] for s in states])




class DFP(TensorwisePreconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        update_freq: int = 1,
        scale_first: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init_scale=init_scale, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: H *= init_scale
            state['H'] = H
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
            if ys != 0 and yy != 0: H *= ys/yy

        # DFP update
        sy = torch.dot(s, y)
        if sy > 1e-10:
            term1 = torch.outer(s, s).div_(sy)
            denom = torch.dot(y, H @ y) #
            if abs(denom) > 1e-10:
                num = H @ torch.outer(y, y) @ H
                term2 = num.div_(denom)
                H.add_(term1.sub_(term2))

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        state['step'] = state.get('step', 0) + 1
        H = state['H']
        return H @ tensor



class Broyden(TensorwisePreconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        update_freq: int = 1,
        scale_first: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init_scale=init_scale, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: H *= init_scale
            state['H'] = H
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
            if ys != 0 and yy != 0: H *= ys/yy

        # Broyden update
        denom = torch.dot(s, H @ y)
        if denom > 1e-10:
            num = (s - H@y).outer(s) @ H
            H.add_(num.div_(denom))

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        state['step'] = state.get('step', 0) + 1
        H = state['H']
        return H @ tensor




class PSB(TensorwisePreconditioner):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        update_freq: int = 1,
        scale_first: bool = True,
        concat_params: bool = True,
        inner: Chainable | None = None,
    ):
        defaults = dict(init_scale=init_scale, tol=tol)
        super().__init__(defaults, uses_grad=False, concat_params=concat_params, update_freq=update_freq, scale_first=scale_first, inner=inner)

    @torch.no_grad
    def update_tensor(self, tensor, param, grad, state, settings):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.get('step', 0)
        init_scale = settings['init_scale']
        tol = settings['tol']

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(init_scale, (int, float)) and init_scale != 1: H *= init_scale
            state['H'] = H
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
            if ys != 0 and yy != 0: H *= ys/yy

        # PSB update
        yy = y.dot(y)
        if yy > 1e-10:
            v = s - H @ y
            term1 = v.outer(y).add_(y.outer(v)).div_(yy)
            term2 = y.outer(y).mul_(y.dot(v).div_(yy**2))
            H.add_(term1.sub_(term2))

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    @torch.no_grad
    def apply_tensor(self, tensor, param, grad, state, settings):
        state['step'] = state.get('step', 0) + 1
        H = state['H']
        return H @ tensor
