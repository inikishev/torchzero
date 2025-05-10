from typing import Any, Literal

import torch

from ...core import Chainable, Module, Preconditioner, TensorwisePreconditioner, Precondition
from ...utils import TensorList


class BFGSInverseUpdateStrategy(TensorwisePreconditioner):
    def __init__(self, tol=1e-10, init_scale: float | Literal['auto'] = 'auto'):
        super().__init__()
        self.init_scale: float | Literal['auto'] = init_scale
        self.tol = tol

    def update_tensor(self, tensor, param, grad, state):
        p = param; g = tensor
        H = state.get('H', None)
        step = state.setdefault('step', 0)
        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: H *= self.init_scale
            state['H'] = H
            state['p_prev'] = p.clone()
            state['g_prev'] = g.clone()
            return

        p_prev = state['p_prev']
        g_prev = state['g_prev']
        s_k: torch.Tensor = p - p_prev
        y_k: torch.Tensor = g - g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().max() <= self.tol:
            return

        if step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: H *= ys/yy

        # BFGS update
        skyk = torch.dot(s_k, y_k)
        if skyk > 1e-10:
            num1 = (skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))
            denom1 = skyk**2
            term1 = num1 / denom1
            num2 = (torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H)
            term2 = num2 / skyk
            H += term1 - term2

        state['p_prev'] = p.clone()
        state['g_prev'] = g.clone()

    def apply_tensor(self, tensor, param, grad, state):
        state['step'] += 1
        H = state['H']
        return H @ tensor



class BFGS(Precondition):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        tol: float = 1e-10,
        update_freq: int = 1,
        scale_first: bool = True,
        tensorwise: bool = False,
        inner: Chainable | None = None,
    ):
        super().__init__(
            BFGSInverseUpdateStrategy(init_scale=init_scale, tol=tol),
            uses_grad=False,
            update_freq=update_freq,
            tensorwise=tensorwise,
            scale_first=scale_first,
            inner=inner,
        )


class SR1InverseUpdateStrategy(TensorwisePreconditioner):
    def __init__(self, eps=1e-8, tol: float = 1e-10, init_scale: float | Literal['auto'] = 1, scale_second:bool=True):
        super().__init__()
        self.eps = eps
        self.init_scale: float | Literal['auto'] = init_scale
        self.tol = tol
        self.scale_second = scale_second

    def update_tensor(self, tensor, param, grad, state):
        p = param; g = tensor
        H = state.get('H', None)
        step = self.counter()

        if H is None:
            H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: H *= self.init_scale
            state['H'] = H
            state['p_prev'] = p.clone()
            state['g_prev'] = g.clone()
            return

        p_prev = state['p_prev']
        g_prev = state['g_prev']
        s_k: torch.Tensor = p - p_prev
        y_k: torch.Tensor = g - g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().max() <= self.tol:
            return

        if step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: H *= ys/yy

        # SR1 update
        z = s_k - H@y_k
        denom = torch.dot(z, y_k)

        # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
        if denom.abs() >= self.eps * torch.linalg.norm(y_k) * torch.linalg.norm(z): # pylint:disable=not-callable
            H += torch.outer(z, z) / denom

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply_tensor(self, tensor, param, grad, state):
        """precondition"""
        self.counter.increment()
        H = state['H']
        if self.scale_second and self.counter() == 2:
            tensor = tensor/max(1, tensor.abs().sum()) # pyright:ignore[reportArgumentType]
        return H @ tensor

class SR1(Precondition):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = 1,
        tol: float = 1e-10,
        eps: float = 1e-8,
        update_freq: int = 1,
        scale_first: bool = True,
        scale_second: bool = True,
        tensorwise: bool = False,
        inner: Chainable | None = None,
    ):
        super().__init__(
            SR1InverseUpdateStrategy(init_scale=init_scale, tol=tol, eps=eps, scale_second=scale_second),
            uses_grad=False,
            update_freq=update_freq,
            tensorwise=tensorwise,
            scale_first=scale_first,
            inner=inner,
        )



class DiagonalBFGSInverseUpdateStrategy(Preconditioner):
    def __init__(self, tol=1e-10, res_beta: float | None = None, H_beta: float | None = None, growth_clip: float | None = None, init_scale: float | Literal['auto'] = 'auto'):
        super().__init__()
        self.init_scale: float | Literal['auto'] = init_scale
        self.res_beta = res_beta
        self.H_beta = H_beta
        self.growth_clip = growth_clip
        self.tol = tol


    def update(self, tensors, params, grads, keys):
        p = TensorList(params); g = TensorList(tensors)
        states = [self.state[k] for k in keys]
        step = self.counter()

        if any('H' not in s for s in states):
            for param, grad, state in zip(params, tensors, states):
                if 'H' not in state:
                    state['H'] = torch.ones_like(param)
                if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: state['H'] *= self.init_scale
                state['p_prev'] = param.clone()
                state['g_prev'] = grad.clone()
            return

        H = TensorList(s['H'] for s in states)
        p_prev = TensorList(s['p_prev'] for s in states)
        g_prev = TensorList(s['g_prev'] for s in states)
        s_k = p - p_prev
        y_k = g - g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().global_max() <= self.tol:
            return

        if step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: H *= ys/yy


        skyk = s_k.dot(y_k)

        # num1 = (skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))
        # denom1 = skyk**2
        # term1 = num1 / denom1
        # num2 = (torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H)
        # term2 = num2 / skyk
        # H += term1 - term2

        if skyk > 1e-10:
            num1 = (skyk + (y_k * H).dot(y_k)) * s_k * s_k
            denom1 = skyk**2
            term1 = num1 / denom1
            z = H * y_k * s_k
            num2 = z + z
            term2 = num2 / skyk
            res = term1 - term2

            if self.res_beta is not None:
                for s,param in zip(states, params):
                    if 'res' not in s:
                        s['res'] = torch.zeros_like(param)
                res = TensorList(s['res'] for s in states)
                res.lerp_(res, 1-self.res_beta)

            if self.growth_clip is not None:
                norm = res.global_vector_norm()

                if 'prev_norm' in self.global_state:
                    prev_norm = self.global_state['prev_norm']
                    allowed_norm = prev_norm * self.growth_clip
                    if norm > allowed_norm:
                        mul = (allowed_norm/norm).clip_(min=1e-5)
                        res.mul_(mul)
                        norm = norm * mul

                self.global_state['prev_norm'] = norm

            if self.H_beta is None: H += res
            else: H.lerp_(H + res, weight=1-self.H_beta)

        p_prev.copy_(p)
        g_prev.copy_(g)

    def apply(self, tensors, params, grads, keys):
        self.counter.increment()
        return TensorList(tensors).mul_([self.state[k]['H'] for k in keys])


class DiagBFGS(Precondition):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        res_beta: float | None = None,
        H_beta: float | None = None,
        growth_clip: float | None = None,
        scale_first: bool = True,
        inner: Chainable | None = None,
    ):
        super().__init__(
            DiagonalBFGSInverseUpdateStrategy(
                res_beta=res_beta,
                H_beta=H_beta,
                init_scale=init_scale,
                growth_clip=growth_clip,
            ),
            uses_grad=False,
            scale_first=scale_first,
            tensorwise=True,
            inner=inner,
        )
