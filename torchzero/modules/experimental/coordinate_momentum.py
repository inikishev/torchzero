import torch

from ...core import TensorTransform
from ...utils import NumberList, TensorList, unpack_states


def coordinate_momentum_(
    tensors: TensorList,
    velocity_: TensorList,
    p: float | NumberList,
):
    """
    sets `velocity_` to p% random values from `tensors`.

    Returns `velocity_`
    """
    mask = tensors.bernoulli_like(p).as_bool()
    velocity_.masked_set_(mask, tensors)
    return velocity_


class CoordinateMomentum(TensorTransform):
    """Maintains a momentum buffer, on each step each value in the buffer has ``p`` chance to be updated with the new value.

    Args:
        p (float, optional): _description_. Defaults to 0.1.
    """
    def __init__(self, p: float = 0.1):
        defaults = dict(p=p)
        super().__init__(defaults)

        self.add_projected_keys("grad", "velocity")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        p = NumberList(s['p'] for s in settings)
        velocity = unpack_states(states, tensors, 'velocity', cls=TensorList)
        return coordinate_momentum_(TensorList(tensors), velocity_=velocity, p=p).clone()
