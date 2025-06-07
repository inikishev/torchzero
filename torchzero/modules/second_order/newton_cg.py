from collections.abc import Callable
from typing import Literal, overload
import warnings
import torch

from ...utils import TensorList, as_tensorlist, generic_zeros_like, generic_vector_norm, generic_numel
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply_transform, Module
from ...utils.linalg.solve import cg

class NewtonCG(Module):
    def __init__(
        self,
        maxiter=None,
        tol=1e-4,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "forward",
        h=1e-3,
        warm_start=False,
        inner: Chainable | None = None,
    ):
        defaults = dict(tol=tol, maxiter=maxiter, reg=reg, hvp_method=hvp_method, h=h, warm_start=warm_start)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        tol = settings['tol']
        reg = settings['reg']
        maxiter = settings['maxiter']
        hvp_method = settings['hvp_method']
        h = settings['h']
        warm_start = settings['warm_start']

        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                with torch.enable_grad():
                    return TensorList(hvp(params, grad, x, retain_graph=True))

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    return TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1])

            elif hvp_method == 'central':
                def H_mm(x):
                    return TensorList(hvp_fd_central(closure, params, x, h=h, normalize=True)[1])

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
        if 'inner' in self.children:
            b = as_tensorlist(apply_transform(self.children['inner'], b, params=params, grads=grad, var=var))

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state(params, 'prev_x', cls=TensorList) # initialized to 0 which is default anyway

        x = cg(A_mm=H_mm, b=as_tensorlist(b), x0_=x0, tol=tol, maxiter=maxiter, reg=reg)
        if warm_start:
            assert x0 is not None
            x0.copy_(x)

        var.update = x
        return var


