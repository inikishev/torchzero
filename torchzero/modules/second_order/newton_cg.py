from collections.abc import Callable
from typing import Literal
import warnings
import torch

from ...utils import TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply, Module

def cg(
    A_mm: Callable[[TensorList], TensorList],
    b: TensorList,
    x0_: TensorList | None,
    tol: float | None,
    maxiter: int | None,
):
    if maxiter is None: maxiter = b.global_numel()
    if x0_ is None: x0_ = b.zeros_like()

    x = x0_
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


class NewtonCG(Module):
    def __init__(self, maxiter=None, tol=1e-3, hvp_method: Literal['forward', 'central','autograd'] = 'forward', h=1e-3, warm_start=False, inner: Chainable | None = None):
        defaults = dict(tol=tol, maxiter=maxiter, hvp_method=hvp_method, h=h, warm_start=warm_start)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, vars):
        params = TensorList(vars.params)
        closure = vars.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        tol = settings['tol']
        maxiter = settings['maxiter']
        hvp_method = settings['hvp_method']
        h = settings['h']
        warm_start = settings['warm_start']

        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            grad = vars.get_grad(create_graph=True)

            def H_mm(x):
                with torch.enable_grad():
                    return TensorList(hvp(params, grad, x, retain_graph=True))

        else:

            with torch.enable_grad():
                grad = vars.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    return TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1])

            elif hvp_method == 'central':
                def H_mm(x):
                    return TensorList(hvp_fd_central(closure, params, x, h=h, normalize=True)[1])

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        b = grad
        if 'inner' in self.children:
            b = as_tensorlist(apply(self.children['inner'], [g.clone() for g in grad], params=params, grads=grad, vars=vars))

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state('prev_x', params=params, cls=TensorList) # initialized to 0 which is default anyway
        x = cg(A_mm=H_mm, b=as_tensorlist(b), x0_=x0, tol=tol, maxiter=maxiter)
        if warm_start:
            assert x0 is not None
            x0.set_(x)

        vars.update = x
        return vars