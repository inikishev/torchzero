from abc import ABC, abstractmethod
from typing import Literal

import torch

from ...core import Chainable, Module, apply
from ...utils import TensorList


class HessianUpdateStrategy(ABC):

    @abstractmethod
    def update(self, p: torch.Tensor, g: torch.Tensor) -> None:
        """update hessian"""


    @abstractmethod
    def apply(self, g: torch.Tensor) -> torch.Tensor:
        """precondition"""

    @abstractmethod
    def clear(self):
        """clear state"""


class QuasiNewton(Module):
    def __init__(self, strategy: HessianUpdateStrategy, scale_first=True, inner: Chainable | None = None):
        defaults = dict(scale_first=scale_first)
        super().__init__(defaults)
        self.strategy = strategy

        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def step(self, vars):
        params = TensorList(vars.params)
        update = TensorList(vars.get_update())
        g = update.to_vec()
        p = params.to_vec()
        self.strategy.update(p, g)
        self.global_state['step'] = self.global_state.get('step', 0) + 1

        if 'inner' in self.children:
            update = TensorList(apply(self.children['inner'], target=update, params=params, grad=vars.grad, vars=vars))
            g = update.to_vec()

        p = self.strategy.apply(g)

        # scale initial step when no preconditioner is available
        if self.settings[vars.params[0]]['scale_first']:
            if self.global_state['step'] == 1:
                p = p / max(1, g.abs().sum()) # pyright:ignore[reportArgumentType]
        vars.update = update.from_vec(p)
        return vars

class BFGSInverseUpdateStrategy(HessianUpdateStrategy):
    def __init__(self, tol=1e-10, init_scale: float | Literal['auto'] = 'auto'):
        self.init_scale: float | Literal['auto'] = init_scale
        self.step = 0
        self.tol = tol
        self.clear()

    def clear(self):
        self.H = None
        self.p_prev = None
        self.g_prev = None

    def update(self, p, g):
        if self.H is None:
            self.H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: self.H *= self.init_scale
            self.p_prev = p.clone()
            self.g_prev = g.clone()
            return

        assert self.p_prev is not None and self.g_prev is not None
        s_k = p - self.p_prev
        y_k = g - self.g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().max() <= self.tol:
            return

        if self.step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: self.H *= ys/yy

        H = self.H

        skyk = torch.dot(s_k, y_k)
        if skyk > 1e-10:
            num1 = (skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))
            denom1 = skyk**2
            term1 = num1 / denom1
            num2 = (torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H)
            term2 = num2 / skyk
            H += term1 - term2

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply(self, g: torch.Tensor) -> torch.Tensor:
        self.step += 1
        return self.H @ g



class BFGS(QuasiNewton):
    def __init__(self, init_scale: float | Literal['auto'] = 'auto', inner: Chainable | None = None):
        super().__init__(BFGSInverseUpdateStrategy(init_scale=init_scale), inner=inner)


class SR1InverseUpdateStrategy(HessianUpdateStrategy):
    def __init__(self, eps=1e-8, tol: float = 1e-10, init_scale: float | Literal['auto'] = 1, scale_second:bool=True):
        self.eps = eps
        self.init_scale: float | Literal['auto'] = init_scale
        self.step = 0
        self.tol = tol
        self.scale_second = scale_second
        self.clear()

    def clear(self):
        self.H = None
        self.p_prev = None
        self.g_prev = None

    def update(self, p, g):
        if self.H is None:
            self.H = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: self.H *= self.init_scale
            self.p_prev = p.clone()
            self.g_prev = g.clone()
            return

        assert self.p_prev is not None and self.g_prev is not None
        s_k = p - self.p_prev
        y_k = g - self.g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().max() <= self.tol:
            return

        if self.step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: self.H *= ys/yy

        H = self.H

        z = s_k - H@y_k
        denom = torch.dot(z, y_k)

        # check as in Nocedal, Wright. “Numerical optimization” 2nd p.146
        if denom.abs() >= self.eps * torch.linalg.norm(y_k) * torch.linalg.norm(z): # pylint:disable=not-callable
            H += torch.outer(z, z) / denom

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply(self, g: torch.Tensor) -> torch.Tensor:
        """precondition"""
        self.step += 1
        if self.scale_second and self.step == 2:
            g = g/max(1, g.abs().sum()) # pyright:ignore[reportArgumentType]
        return self.H @ g

class SR1(QuasiNewton):
    def __init__(self, eps=1e-8, init_scale: float | Literal['auto'] = 1, scale_second=True, inner: Chainable | None = None):
        super().__init__(SR1InverseUpdateStrategy(eps=eps, init_scale=init_scale, scale_second=scale_second), inner=inner,)




class DiagonalBFGSInverseUpdateStrategy(HessianUpdateStrategy):
    def __init__(self, tol=1e-10, res_beta: float | None = None, H_beta: float | None = None, growth_clip: float | None = None, init_scale: float | Literal['auto'] = 'auto'):
        self.init_scale: float | Literal['auto'] = init_scale
        self.step = 0
        self.res_beta = res_beta
        self.H_beta = H_beta
        self.growth_clip = growth_clip
        self.tol = tol
        self.res = None
        self.prev_res = None
        self.clear()

    def clear(self):
        self.H = None
        self.p_prev = None
        self.g_prev = None

    def update(self, p, g):
        if self.H is None:
            self.H = torch.ones_like(p)
            if isinstance(self.init_scale, (int, float)) and self.init_scale != 1: self.H *= self.init_scale
            self.p_prev = p.clone()
            self.g_prev = g.clone()
            return

        assert self.p_prev is not None and self.g_prev is not None
        s_k = p - self.p_prev
        y_k = g - self.g_prev

        # tolerance on gradient difference to avoid exploding after converging
        if y_k.abs().max() <= self.tol:
            return

        if self.step == 1 and self.init_scale == 'auto':
            ys = y_k.dot(s_k)
            yy = y_k.dot(y_k)
            if ys != 0 and yy != 0: self.H *= ys/yy

        H = self.H

        skyk = torch.dot(s_k, y_k)

        # num1 = (skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))
        # denom1 = skyk**2
        # term1 = num1 / denom1
        # num2 = (torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H)
        # term2 = num2 / skyk
        # H += term1 - term2

        if skyk > 1e-10:
            num1 = (skyk + torch.dot(y_k * H, y_k)) * s_k * s_k
            denom1 = skyk**2
            term1 = num1 / denom1
            z = H * y_k * s_k
            num2 = z + z
            term2 = num2 / skyk
            res = term1 - term2

            if self.res_beta is not None:
                if self.res is None: self.res = torch.zeros_like(res)
                self.res.lerp_(res, 1-self.res_beta)
                res = self.res

            if self.growth_clip is not None:
                if self.prev_res is not None:
                    prev_norm = torch.linalg.norm(self.prev_res) # pylint:disable=not-callable
                    allowed_norm = prev_norm * self.growth_clip
                    norm = torch.linalg.norm(res) # pylint:disable=not-callable
                    if norm > allowed_norm:
                        res.mul_((allowed_norm/norm).clip(min=1e-5))
                self.prev_res = res.clone()


            if self.H_beta is None: H += res
            else: H.lerp_(H + res, weight=1-self.H_beta)

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply(self, g: torch.Tensor) -> torch.Tensor:
        self.step += 1
        assert self.H is not None
        return self.H * g



class DiagonalBFGS(QuasiNewton):
    def __init__(
        self,
        init_scale: float | Literal["auto"] = "auto",
        res_beta: float | None = None,
        H_beta: float | None = None,
        growth_clip: float | None = None,
        inner: Chainable | None = None,
    ):
        super().__init__(
            DiagonalBFGSInverseUpdateStrategy(
                res_beta=res_beta,
                H_beta=H_beta,
                init_scale=init_scale,
                growth_clip=growth_clip,
            ),
            inner=inner,
        )
