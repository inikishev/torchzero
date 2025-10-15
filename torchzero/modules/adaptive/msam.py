from typing import Literal

import torch

from ...core import Chainable, Module,  Transform, TensorTransform, step, Objective
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states, generic_ne
from ..opt_utils import ema_
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
    objective: Objective | None = None,
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

    denom = velocity_.global_vector_norm() / rho
    denom = denom.clip(min=torch.finfo(tensors[0].dtype).tiny * 2)
    vn = velocity_ / denom

    mom_ = nag_ if nesterov else ema_
    velocity_ = mom_(tensors, velocity_, momentum, dampening=0, lerp=lerp)

    denom = velocity_.global_vector_norm() / rho
    denom = denom.clip(min=torch.finfo(tensors[0].dtype).tiny * 2)
    v1n = velocity_ / denom

    if inner is not None:
        assert objective is not None and inner is not None
        inner_update = TensorList(step(objective, inner).get_updates())

    else:
        assert lr is not None
        inner_update = velocity_ * lr

    update = inner_update.add_(v1n).sub_(vn)

    if generic_ne(weight_decay, 0):
        wd = (params + vn).mul_(weight_decay)
        update.add_(wd)

    return update

class MSAMMomentum(TensorTransform):
    """Momentum-SAM from https://arxiv.org/pdf/2401.12033.

    This implementation expresses the update rule as function of gradient. This way it can be used as a drop-in
    replacement for momentum strategies in other optimizers.

    To combine MSAM with other optimizers in the way done in the official implementation,
    e.g. to make Adam_MSAM, use ``tz.m.MSAMObjective`` module.

    Note
        MSAM has a learning rate hyperparameter that can't really be removed from the update rule.
        To avoid compounding learning rate mofications, remove the ``tz.m.LR`` module if you had it.

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

    ### Examples:

    MSAM

    ```python

    opt = tz.Optimizer(
        model.parameters(),
        tz.m.MSAM(1e-3)
    )
    ```

    Adam with MSAM instead of exponential average. Note that this is different from Adam_MSAM.
    To make Adam_MSAM and such, use the ``tz.m.MSAMObjective`` module.

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.RMSprop(0.999, inner=tz.m.MSAM(1e-3)),
        tz.m.Debias(0.9, 0.999),
    )
    ```
    """

    def __init__(self, lr: float, momentum:float=0.9, rho:float=0.3,  weight_decay:float=0, nesterov=False, lerp=False,):
        defaults = dict(lr = lr, momentum=momentum, rho=rho, nesterov=nesterov, lerp=lerp, weight_decay=weight_decay)
        super().__init__(defaults, uses_grad=False)

        self.add_projected_keys("grad", "velocity")

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        velocity = unpack_states(states, tensors, 'velocity', cls=TensorList)
        fs = settings[0]

        lr, momentum, rho, weight_decay = unpack_dicts(settings, 'lr','momentum','rho','weight_decay', cls=NumberList)

        return msam_(
            TensorList(tensors),
            params=TensorList(params),
            velocity_=velocity,
            momentum=momentum,
            lr=lr,
            rho=rho,
            weight_decay=weight_decay,
            nesterov=fs['nesterov'],
            lerp=fs['lerp'],

            # inner args
            inner=None,
            objective=None,
        )


class MSAM(Transform):
    """Momentum-SAM from https://arxiv.org/pdf/2401.12033.

    Note:
        Please make sure to place ``tz.m.LR`` inside the ``modules`` argument. For example,
        ``tz.m.MSAMObjective([tz.m.Adam(), tz.m.LR(1e-3)])``. Putting LR after MSAM will lead
        to an incorrect update rule.

    Args:
        modules (Chainable): modules that will optimize the MSAM objective. Make sure ``tz.m.LR`` is one of them.
        momentum (float, optional): momentum (beta). Defaults to 0.9.
        rho (float, optional): perturbation strength. Defaults to 0.3.
        nesterov (bool, optional): whether to use nesterov momentum formula. Defaults to False.
        lerp (bool, optional):
            whether to use linear interpolation, if True, MSAM momentum becomes similar to exponential moving average.
            Defaults to False.

    Examples:
    AdamW-MSAM

    ```py
    opt = tz.Optimizer(
        bench.parameters(),
        tz.m.MSAMObjective(
            [tz.m.Adam(), tz.m.WeightDecay(1e-3), tz.m.LR(1e-3)],
            rho=1.
        )
    )
    ```
    """
    def __init__(self, modules: Chainable, momentum:float=0.9, rho:float=0.3, weight_decay:float=0, nesterov=False, lerp=False):
        defaults = dict(momentum=momentum, rho=rho, weight_decay=weight_decay, nesterov=nesterov, lerp=lerp)
        super().__init__(defaults)

        self.set_child('modules', modules)
        self.add_projected_keys("grad", "velocity")


    @torch.no_grad
    def apply_states(self, objective, states, settings):
        velocity = unpack_states(states, objective.params, 'velocity', cls=TensorList)
        fs = settings[0]

        momentum, rho, weight_decay = unpack_dicts(settings, 'momentum', 'rho', 'weight_decay', cls=NumberList)

        return msam_(
            TensorList(objective.get_updates()),
            params=TensorList(objective.params),
            velocity_=velocity,
            momentum=momentum,
            lr=None,
            rho=rho,
            weight_decay=weight_decay,
            nesterov=fs['nesterov'],
            lerp=fs['lerp'],

            # inner args
            inner=self.children["modules"],
            objective=objective,
        )
