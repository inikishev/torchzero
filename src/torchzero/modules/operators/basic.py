import torch

from ...core import OptimizerModule, _get_loss


class Clone(OptimizerModule):
    """Clones the ascent direction.
    If this is the first module in a chain, this will clone the gradient.
    That is useful to avoid directly modifying the gradient."""
    def __init__(self):
        super().__init__({})

    @torch.no_grad
    def _update(self, state, ascent_direction):
        return ascent_direction.clone()


class Lambda(OptimizerModule):
    """Applies a function to the ascent direction. The function must take a TensorList as the argument."""
    def __init__(self, f):
        super().__init__({})
        self.f = f

    @torch.no_grad()
    def _update(self, state, ascent_direction):
        return self.f(ascent_direction)

class Reciprocal(OptimizerModule):
    def __init__(self,):
        super().__init__({})

    @torch.no_grad()
    def _update(self, state, ascent_direction):
        ascent_direction.reciprocal_()
        return ascent_direction

class NanToNum(OptimizerModule):
    def __init__(self, nan=None, posinf=None, neginf=None):
        super().__init__({})
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    @torch.no_grad()
    def _update(self, state, ascent_direction):
        ascent_direction.nan_to_num_(self.nan, self.posinf, self.neginf)
        return ascent_direction