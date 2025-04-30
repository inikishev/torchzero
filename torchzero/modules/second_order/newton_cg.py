from collections.abc import Callable
from typing import Literal

import torch

from ...utils import TensorList
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward
from ..grad_approximation import GradMaker, GradTarget
from ...core import Chainable, apply

def cg(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0: TensorList | None,
    tol: float | None,
    maxiter: int | None,
):
    if maxiter is None: maxiter = b.global_numel()
    if x0 is None: x0 = b.zeros_like()

    x = x0
    residual = b - A_mm(x)
    p = residual.clone() # search direction
    r_norm = residual.global_vector_norm()
    if tol is not None and r_norm < tol: return x
    k = 0

    while True:
        Ap = A_mm(p)
        step_size = (r_norm**2) / p.dot(Ap)
        x += step_size * p # Update solution
        residual -= step_size * Ap # Update residual
        new_r_norm = residual.global_vector_norm()

        k += 1
        if tol is not None and new_r_norm <= tol: return x
        if maxiter is not None and k >= maxiter: return x

        beta = (new_r_norm**2) / (r_norm**2)
        p = residual + beta*p
        r_norm = new_r_norm


class NewtonCG(GradMaker):
    def __init__(self, tol=1e-3, maxiter=None, hvp_method: Literal['forward', 'central','autograd'] = 'autograd', h=1e-3, warm_start=False, target: GradTarget = 'update', inner: Chainable | None = None):
        defaults = dict(tol=tol, maxiter=maxiter, hvp_method=hvp_method, h=h, warm_start=warm_start)
        super().__init__(defaults, target=target)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def approximate(self, closure, params, loss, vars):
        params = TensorList(params)
        settings = self.settings[params[0]]
        tol = settings['tol']
        maxiter = settings['maxiter']
        hvp_method = settings['hvp_method']
        h = settings['h']
        warm_start = settings['warm_start']

        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            for p in params: p.grad = None
            with torch.enable_grad():
                loss = closure(False)
                loss.backward(create_graph=True)
            grad = params.grad

            def H_mm(x):
                with torch.enable_grad():
                    return TensorList(hvp(params, grad.clone(), x, retain_graph=True))

        else:

            with torch.enable_grad():
                loss = closure(False)
                loss.backward(retain_graph=True)
                grad = params.grad

            if hvp_method == 'forward':
                def H_mm(x):
                    return TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True))

            elif hvp_method == 'central':
                def H_mm(x):
                    return TensorList(hvp_fd_central(closure, params, x, h=h, normalize=True))

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        if 'inner' in self.children:
            grad = apply(self.children['inner'], grad, params=params, grad=grad, vars=vars)

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state('prev_x', params=params, cls=TensorList) # initialized to 0 which is default anyway
        x = cg(A_mm=H_mm, b=TensorList(grad), x0=x0, tol=tol, maxiter=maxiter)
        if warm_start:
            assert x0 is not None
            x0.set_(x)

        return x, loss, loss