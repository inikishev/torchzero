from typing import Literal
from collections.abc import Callable
import torch

from ...core import Module, Target, Transform, Chainable, apply_transform
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_forward, hvp_fd_central

def curveball(
    tensors: TensorList,
    z_: TensorList,
    Hz: TensorList,
    momentum: float | NumberList,
    precond_lr: float | NumberList,
):
    """returns z_, clone it!!! (no just negate it)"""
    delta = Hz + tensors
    z_.mul_(momentum).sub_(delta.mul_(precond_lr)) # z ← ρz − βΔ
    return z_


class CurveBall(Module):
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
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h: float = 1e-3,
        reg: float = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(precond_lr=precond_lr, momentum=momentum, hvp_method=hvp_method, h=h, reg=reg)
        super().__init__(defaults)

        if inner is not None: self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):

        params = var.params
        settings = self.settings[params[0]]
        hvp_method = settings['hvp_method']
        h = settings['h']

        precond_lr, momentum, reg = self.get_settings(params, 'precond_lr', 'momentum', 'reg', cls=NumberList)


        closure = var.closure
        assert closure is not None

        z, Hz = self.get_state(params, 'z', 'Hz', cls=TensorList)

        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)
            Hvp = hvp(params, grad, z)

        elif hvp_method == 'forward':
            loss, Hvp = hvp_fd_forward(closure, params, z, h=h, g_0=var.get_grad(), normalize=True)

        elif hvp_method == 'central':
            loss, Hvp = hvp_fd_central(closure, params, z, h=h, normalize=True)

        else:
            raise ValueError(hvp_method)


        Hz.set_(Hvp + z*reg)


        update = var.get_update()
        if 'inner' in self.children:
            update = apply_transform(self.children['inner'], update, params, grads=var.grad, var=var)

        z = curveball(TensorList(update), z, Hz, momentum=momentum, precond_lr=precond_lr)
        var.update = z.neg()

        return var