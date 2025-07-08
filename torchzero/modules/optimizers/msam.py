from typing import Literal

import torch

from ...core import Chainable, Module, Target, Transform, apply_transform
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states, generic_ne
from ..functional import ema_
from ..momentum.momentum import nag_


def msam_(
    tensors: TensorList,
    params: TensorList,
    velocity_: TensorList,
    momentum: float | NumberList,
    lr: NumberList | None,
    rho: float | NumberList,
    weight_decay: float | NumberList,
    nesterov: bool = False,
    lerp: bool = False,

    # inner args
    inner: Module | None = None,
    grads: list[torch.Tensor] | None = None,
):
    # weights w and wh, momentum μ, perturbation strength ρ
    # w = wh + rho * v / ||v||
    # v1 = μv + g
    # w1 = w - lr*v1
    # wh1 = w1 - rho * v1 / ||v1||

    # w1 = wh + rho * v / ||v|| - lr*v1
    # vn = rho * v / ||v||
    # v1n = rho * v1 / ||v1||
    # wh1 = wh + vn - lr*v1 - v1n

    # the update is
    # vn - lr*v1 - v1n

    # we track ascent direction so it becomes lr*v1 + v1n - vn

    # can't really decouple it from lr
    # but at least it is now expressed as function of g

    denom = (velocity_.global_vector_norm() / rho).clip(min=1e-8)
    vn = velocity_ / denom

    mom_ = nag_ if nesterov else ema_
    velocity_ = mom_(tensors, velocity_, momentum, dampening=0, lerp=lerp)

    denom = (velocity_.global_vector_norm() / rho).clip(min=1e-8)
    v1n = velocity_ / denom

    if inner is not None:
        assert params is not None
        inner_update = TensorList(apply_transform(inner, tensors, params=params, grads=grads))

    else:
        assert lr is not None
        inner_update = velocity_ * lr

    update = inner_update.add_(v1n).sub_(vn)

    if generic_ne(weight_decay, 0):
        wd = (params + vn).mul_(weight_decay)
        update.add_(wd)

    return update

class MSAM(Transform):
    """Momentum-SAM from https://arxiv.org/pdf/2401.12033.

    This implementation expresses the update rule as function of gradient. This way it can be used as a drop-in
    replacement for momentum strategies in other optimizers.

    To combine MSAM with other optimizers in the way done in the official implementation,
    e.g. to make Adam_MSAM, use :code:`tz.m.MSAMObjective` module.

    .. note::
        MSAM has a learning rate hyperparameter that can't really be removed from the update rule.
        To avoid compounding learning rate mofications, remove the :code:`tz.m.LR` module if you had it.

    Args:
        lr (float): learning rate. Adding this module adds support for learning rate schedulers.
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        rho (float, optional): perturbation strength. Defaults to 0.3.
        weight_decay (float, optional):
            weight decay. It is applied to perturbed parameters, so it is differnet
            from applying :code:`tz.m.WeightDecay` after MSAM. Defaults to 0.
        nesterov (bool, optional): whether to use nesterov momentum formula. Defaults to False.
        lerp (bool, optional):
            whether to use linear interpolation, if True, this becomes similar to exponential moving average. Defaults to False.

    Examples:
        MSAM

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.MSAM(1e-3)
            )

        Adam with MSAM instead of exponential average. Note that this is different from Adam_MSAM.
        To make Adam_MSAM and such, use the :code:`tz.m.MSAMObjective` module.

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.RMSprop(0.999, inner=tz.m.MSAM(1e-3)),
                tz.m.Debias(0.9, 0.999),
            )
    """
    USES_LR = True
    def __init__(self, lr: float, momentum:float=0.9, rho:float=0.3,  weight_decay:float=0, nesterov=False, lerp=False,):
        defaults = dict(momentum=momentum,rho=rho, nesterov=nesterov, lerp=lerp, weight_decay=weight_decay)
        if self.USES_LR: defaults['lr'] = lr
        super().__init__(defaults, uses_grad=False)

    @torch.no_grad
    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        velocity = unpack_states(states, tensors, 'velocity', cls=TensorList)
        s = self.settings[params[0]]
        lerp = s['lerp']
        nesterov = s['nesterov']

        if self.USES_LR:
            lr, momentum, rho, weight_decay = unpack_dicts(settings, 'lr','momentum','rho','weight_decay', cls=NumberList)

        else:
            lr=None
            momentum,rho,weight_decay = unpack_dicts(settings, 'momentum','rho','weight_decay', cls=NumberList)

        return msam_(
            TensorList(tensors),
            params=TensorList(params),
            velocity_=velocity,
            momentum=momentum,
            lr=lr,
            rho=rho,
            weight_decay=weight_decay,
            nesterov=nesterov,
            lerp=lerp,

            # inner args
            inner=self.children.get("modules", None),
            grads=grads,
        )


class MSAMObjective(MSAM):
    """Momentum-SAM from https://arxiv.org/pdf/2401.12033.

    .. note::
        Please make sure to place :code:`tz.m.LR` inside the :code:`modules` argument. For example,
        :code:`tz.m.MSAMObjective([tz.m.Adam(), tz.m.LR(1e-3)])`. Putting LR after MSAM will lead
        to an incorrect update rule.

    Args:
        modules (Chainable): modules that will optimizer the MSAM objective. Make sure :code:`tz.m.LR` is one of them.
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        rho (float, optional): perturbation strength. Defaults to 0.3.
        nesterov (bool, optional): whether to use nesterov momentum formula. Defaults to False.
        lerp (bool, optional):
            whether to use linear interpolation, if True, MSAM momentum becomes similar to exponential moving average.
            Defaults to False.

    Examples:
        AdamW-MSAM

        .. code-block:: python

            opt = tz.Modular(
                bench.parameters(),
                tz.m.MSAMObjective(
                    [tz.m.Adam(), tz.m.WeightDecay(1e-3), tz.m.LR(1e-3)],
                    rho=1.
                )
            )
    """
    USES_LR = False
    def __init__(self, modules: Chainable, momentum:float=0.9, rho:float=0.3, weight_decay:float=0, nesterov=False, lerp=False):
        super().__init__(lr=0, momentum=momentum, rho=rho, weight_decay=weight_decay, nesterov=nesterov, lerp=lerp)
        self.set_child('modules', modules)

