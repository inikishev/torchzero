from abc import ABC, abstractmethod
import torch
from ...core import Module, Chainable, apply
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

class BFGSInverseUpdateStrategy(HessianUpdateStrategy):
    def __init__(self): self.clear()

    def clear(self):
        self.B_inv = None
        self.p_prev = None
        self.g_prev = None

    def update(self, p, g):
        if self.B_inv is None:
            self.B_inv = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            self.p_prev = p.clone()
            self.g_prev = g.clone()
            return

        assert self.p_prev is not None and self.g_prev is not None
        s_k = p - self.p_prev
        y_k = g - self.g_prev
        H = self.B_inv # for some reason hessian is B and inverse of hessian is H in all references

        skyk = torch.dot(s_k, y_k)
        if skyk > 1e-10:
            H += ((skyk + (y_k @ H @ y_k)) * (torch.outer(s_k, s_k))) / (skyk**2) - \
                ((torch.outer(H @ y_k, s_k) + torch.outer(s_k, y_k) @ H) / skyk)

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply(self, g: torch.Tensor) -> torch.Tensor:
        return self.B_inv @ g


class QuasiNewton(Module):
    def __init__(self, strategy: HessianUpdateStrategy, scale_first=True, inner: Chainable | None = None):
        defaults = dict(scale_first=scale_first)
        super().__init__(defaults)
        self.strategy = strategy

        if inner is not None: self.set_child('inner', inner)

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

class BFGS(QuasiNewton):
    def __init__(self):
        super().__init__(BFGSInverseUpdateStrategy())


class SR1InverseUpdateStrategy(HessianUpdateStrategy):
    def __init__(self): self.clear()

    def clear(self):
        self.B_inv = None
        self.p_prev = None
        self.g_prev = None

    def update(self, p, g):
        if self.B_inv is None:
            self.B_inv = torch.eye(p.size(0), device=p.device, dtype=p.dtype)
            self.p_prev = p.clone()
            self.g_prev = g.clone()
            return

        assert self.p_prev is not None and self.g_prev is not None
        s_k = p - self.p_prev
        y_k = g - self.g_prev
        H = self.B_inv # for some reason hessian is B and inverse of hessian is H in all references

        z = s_k - H@y_k
        H += torch.outer(z, z) / (torch.dot(z, y_k))

        self.p_prev = p.clone()
        self.g_prev = g.clone()

    def apply(self, g: torch.Tensor) -> torch.Tensor:
        """precondition"""
        return self.B_inv @ g

class SR1(QuasiNewton):
    def __init__(self):
        super().__init__(SR1InverseUpdateStrategy())
