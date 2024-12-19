import typing
import torch

from ...core import OptimizerModule

class ReduceOutwardLR(OptimizerModule):
    """
    When update sign matches weight sign, the learning rate for that weight is multiplied by `mul`.

    This means updates that move weights towards zero have higher learning rates.

    .. warning::
        If `use_grad` is True and you use this after modules that estimate gradients, e.g. FDM,
        they need to have `make_closure` set to True so that they write to `grad` attribute.
    """
    def __init__(self, mul = 0.5, use_grad=False):
        defaults = dict(mul = mul)
        super().__init__(defaults)

        self.use_grad = use_grad

    @torch.no_grad
    def _update(self, state, ascent):
        params = self.get_params()
        mul = self.get_group_key('mul')

        if self.use_grad: cur = state.maybe_compute_grad_(params)
        else: cur = ascent

        # mask of weights where sign matches with update sign, multiplied by `mul`.
        mask = (params * cur).le_(0).mul_(mul)
        ascent.mul_(mask)

        return ascent

