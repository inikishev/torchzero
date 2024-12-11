import torch

from ...core import OptimizerModule


class LR(OptimizerModule):
    """Multiplies the ascent direction by the learning rate."""
    def __init__(self, lr = 1e-3):
        defaults = dict(lr = lr)
        super().__init__(defaults)

    @torch.no_grad
    def _update(self, state, ascent):
        # multiply ascent direction by lr in-place
        lr = self.get_group_key('lr')
        ascent *= lr
        return ascent
