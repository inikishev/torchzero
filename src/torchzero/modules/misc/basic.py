from collections.abc import Callable, Iterable

import torch

from torchzero.tensorlist import TensorList

from ...core import OptimizerModule

class LR(OptimizerModule):
    """Multiplies update by the learning rate."""
    IS_LR_MODULE = True
    def __init__(self, lr = 1e-3):
        defaults = dict(lr = lr)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        # multiply ascent direction by lr in-place
        lr = self.get_group_key('lr')
        ascent *= lr
        return ascent

class Alpha(OptimizerModule):
    """Multiplies update by the learning rate, won't get picked up by learning rate schedulers."""
    def __init__(self, alpha = 1e-3):
        defaults = dict(alpha = alpha)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        # multiply ascent direction by lr in-place
        lr = self.get_group_key('alpha')
        ascent *= lr
        return ascent

class Clone(OptimizerModule):
    """Clones the update. Some modules update ascent in-place, so this may be
    useful if you need to preserve it."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent.clone()

class Identity(OptimizerModule):
    """Does nothing."""
    def __init__(self, *args, **kwargs):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent): return ascent

class Lambda(OptimizerModule):
    """Applies a function to the ascent direction.
    The function must take a TensorList as the argument, and return the modified tensorlist.

    Args:
        f (Callable): function
    """
    def __init__(self, f: Callable[[TensorList], TensorList]):
        super().__init__({})
        self.f = f

    @torch.no_grad()
    def _update(self, state, ascent): return self.f(ascent)

class Grad(OptimizerModule):
    """Uses gradient as the update. This is useful for chains."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        ascent = state.ascent = state.maybe_compute_grad_(self.get_params())
        return ascent

class Zeros(OptimizerModule):
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent):
        return ascent.zeros_like()

class Fill(OptimizerModule):
    def __init__(self, value):
        super().__init__({"value": value})

    @torch.no_grad
    def _update(self, state, ascent):
        return ascent.fill(self.get_group_key('value'))