from operator import itemgetter
from functools import partial

import torch

from ...core import Module, Target, Transform, apply_transform, Chainable
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states
from ..functional import (
    debias, debiased_step_size,
    ema_,
    sqrt_ema_sq_,
)
from ..step_size.lr import lazy_lr
from ..momentum.experimental import sqrt_nag_ema_sq_
from ..momentum.momentum import nag_


def mars_correction_(
    tensors_: TensorList,
    prev_: TensorList,
    beta: float | NumberList,
    scaling: float | NumberList,
    max_norm: float | NumberList |  None,
):
    dg = (tensors_ - prev_).mul_(scaling * beta / (1-beta))
    prev_.copy_(tensors_)

    c = tensors_.add_(dg)
    if max_norm is not None:
        c.clip_norm_(max=max_norm, tensorwise=False)

    return c

class MARSCorrection(Transform):
    """MARS variance reduction correction.

    Place any other momentum-based optimizer after this,
    make sure :code:`beta` parameter matches with momentum in the optimizer.

    Args:
        beta (float, optional): use the same beta as you use in the momentum module. Defaults to 0.9.
        scaling (float, optional): controls the scale of gradient correction in variance reduction. Defaults to 0.025.
        max_norm (float, optional): clips norm of corrected gradients, None to disable. Defaults to 1.

    Examples:
        Mars-AdamW

        .. code-block:: python

            optimizer = tz.Modular(
                model.parameters(),
                tz.m.MARSCorrection(beta=0.95),
                tz.m.Adam(beta1=0.95, beta2=0.99),
                tz.m.WeightDecay(1e-3),
                tz.m.LR(0.1)
            )

        Mars-Lion

        .. code-block:: python

            optimizer = tz.Modular(
                model.parameters(),
                tz.m.MARSCorrection(beta=0.9),
                tz.m.Lion(beta1=0.9),
                tz.m.LR(0.1)
            )

    """
    def __init__(
        self,
        beta: float = 0.9,
        scaling: float = 0.025,
        max_norm: float | None = 1,
    ):
        defaults=dict(beta=beta, scaling=scaling, max_norm=max_norm)
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        prev = unpack_states(states, tensors, 'prev', init=tensors, cls=TensorList)
        beta, scaling = unpack_dicts(settings, 'beta', 'scaling', cls=NumberList)
        max_norm = settings[0]['max_norm']

        return mars_correction_(
            tensors_=TensorList(tensors),
            prev_=prev,
            beta=beta,
            scaling=scaling,
            max_norm=max_norm,
        )