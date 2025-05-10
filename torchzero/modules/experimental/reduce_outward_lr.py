import torch

from ...core import Target, Transform
from ...utils import TensorList

class ReduceOutwardLR(Transform):
    """
    When update sign matches weight sign, the learning rate for that weight is multiplied by `mul`.

    This means updates that move weights towards zero have higher learning rates.
    """
    def __init__(self, mul = 0.5, use_grad=False, invert=False, target: Target = 'update'):
        defaults = dict(mul=mul, use_grad=use_grad, invert=invert)
        super().__init__(defaults, uses_grad=use_grad, target=target)

    @torch.no_grad
    def transform(self, tensors, params, grads, vars):
        params = TensorList(params)
        tensors = TensorList(tensors)

        mul = self.get_settings('mul', params=params)
        s = self.settings[params[0]]
        use_grad = s['use_grad']
        invert = s['invert']

        if use_grad: cur = vars.get_grad()
        else: cur = tensors

        # mask of weights where sign matches with update sign (minus ascent sign), multiplied by `mul`.
        if invert: mask = (params * cur) > 0
        else: mask = (params * cur) < 0

        tensors.masked_set_(mask, tensors*mul)

        return tensors
