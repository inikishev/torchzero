from collections.abc import Callable
from abc import ABC, abstractmethod
import torch
from ...core import Module
from ...core import Chainable

class HomotopyBase(Module):
    def __init__(self, defaults: dict | None = None):
        super().__init__(defaults)

    @abstractmethod
    def loss_transform(self, loss: torch.Tensor) -> torch.Tensor:
        """transform the loss"""

    @torch.no_grad
    def step(self, var):
        if var.loss is not None:
            var.loss = self.loss_transform(var.loss)

        closure = var.closure
        if closure is None: raise RuntimeError("SquareHomotopy requires closure")

        def homotopy_closure(backward=True):
            if backward:
                with torch.enable_grad():
                    loss = self.loss_transform(closure(False))
                    grad = torch.autograd.grad(loss, var.params, allow_unused=True)
                    for p,g in zip(var.params, grad):
                        p.grad = g
            else:
                loss = self.loss_transform(closure(False))

            return loss

        var.closure = homotopy_closure
        return var

class SquareHomotopy(HomotopyBase):
    def __init__(self): super().__init__()
    def loss_transform(self, loss): return loss.square().copysign(loss)

class SqrtHomotopy(HomotopyBase):
    def __init__(self): super().__init__()
    def loss_transform(self, loss): return (loss+1e-12).sqrt()

class ExpHomotopy(HomotopyBase):
    def __init__(self): super().__init__()
    def loss_transform(self, loss): return loss.exp()

class LogHomotopy(HomotopyBase):
    def __init__(self): super().__init__()
    def loss_transform(self, loss): return (loss+1e-12).log()

class LambdaHomotopy(HomotopyBase):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        defaults = dict(fn=fn)
        super().__init__(defaults)

    def loss_transform(self, loss): return self.defaults['fn'](loss)
