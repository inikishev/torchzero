from typing import Literal

import torch

from ...core import Chainable, Transform, step, HVPMethod
from ...utils import NumberList, TensorList, unpack_dicts, unpack_states


def curveball(
    tensors: TensorList,
    z_: TensorList,
    Hzz: TensorList,
    momentum: float | NumberList,
    precond_lr: float | NumberList,
):
    """returns z_, clone it!!! (no just negate it)"""
    delta = Hzz + tensors
    z_.mul_(momentum).sub_(delta.mul_(precond_lr)) # z ← ρz − βΔ
    return z_


class CurveBall(Transform):
    """CurveBall method from https://arxiv.org/pdf/1805.08095#page=4.09.

    For now this implementation does not include automatic ρ, α and β hyper-parameters in closed form, therefore it is expected to underperform compared to official implementation (https://github.com/jotaf98/pytorch-curveball/tree/master) so I moved this to experimental.

    Args:
        precond_lr (float, optional): learning rate for updating preconditioned gradients. Defaults to 1e-3.
        momentum (float, optional): decay rate for preconditioned gradients. Defaults to 0.9.
        hvp_method (str, optional): how to calculate hessian vector products. Defaults to "autograd".
        h (float, optional): finite difference step size for when hvp_method is set to finite difference. Defaults to 1e-3.
        reg (float, optional): hessian regularization. Defaults to 1.
        inner (Chainable | None, optional): Inner modules. Defaults to None.
    """
    def __init__(
        self,
        precond_lr: float=1e-3,
        momentum: float=0.9,
        hvp_method: HVPMethod = "autograd",
        h: float = 1e-3,
        reg: float = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(precond_lr=precond_lr, momentum=momentum, hvp_method=hvp_method, h=h, reg=reg)
        super().__init__(defaults)

        self.set_child('inner', inner)

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        params = objective.params
        fs = settings[0]
        hvp_method = fs['hvp_method']
        h = fs['h']

        precond_lr, momentum, reg = unpack_dicts(settings, 'precond_lr', 'momentum', 'reg', cls=NumberList)

        closure = objective.closure
        assert closure is not None

        z, Hz = unpack_states(states, params, 'z', 'Hz', cls=TensorList)
        Hz, _ = objective.hessian_vector_product(z, rgrad=None, at_x0=True, hvp_method=hvp_method, h=h)

        Hz = TensorList(Hz)
        Hzz = Hz.add_(z * reg)

        objective = self.inner_step("inner", objective, must_exist=False)
        updates = objective.get_updates()

        z = curveball(TensorList(updates), z, Hzz, momentum=momentum, precond_lr=precond_lr)
        objective.updates = z.neg()

        return objective