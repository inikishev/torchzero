import torch

from ...core import  TensorTransform
from ...utils import TensorList, unpack_states, unpack_dicts

class ReduceOutwardLR(TensorTransform):
    """When update sign matches weight sign, the learning rate for that weight is multiplied by `mul`.

    This means updates that move weights towards zero have higher learning rates.

    Warning:
        This sounded good but after testing turns out it sucks.
    """
    def __init__(self, mul = 0.5, use_grad=False, invert=False):
        defaults = dict(mul=mul, use_grad=use_grad, invert=invert)
        super().__init__(defaults, uses_grad=use_grad)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        params = TensorList(params)
        tensors = TensorList(tensors)

        mul = [s['mul'] for s in settings]
        s = settings[0]
        use_grad = self._uses_grad
        invert = s['invert']

        if use_grad: cur = grads
        else: cur = tensors
        assert cur is not None

        # mask of weights where sign matches with update sign (minus ascent sign), multiplied by `mul`.
        if invert: mask = (params * cur) > 0
        else: mask = (params * cur) < 0

        tensors.masked_set_(mask, tensors*mul)

        return tensors
