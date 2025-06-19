from collections.abc import Callable
from typing import Literal, overload
import warnings
import torch

from ...utils import TensorList, as_tensorlist, generic_zeros_like, generic_vector_norm, generic_numel
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply_transform, Module
from ...utils.linalg.solve import cg

class NewtonCG(Module):
    """Newton's method with a matrix-free conjugate gradient solver. This doesn't require the hessian but it calculates hessian-vector products. Those can be obtained with autograd or finite difference method depending on the value of the :coed:`hvp_method` argument.

    Args:
        maxiter (int | None, optional):
            maximum number of iterations. By default this is set to the number of dimensions
            in the objective function, which is supposed to be enough for conjugate gradient
            to have guaranteed convergence. Setting this to a small value can still generate good enough directions.
            Defaults to None.
        tol (float, optional): relative tolerance for conjugate gradient solver. Defaults to 1e-4.
        reg (float, optional): regularization parameter. Defaults to 1e-8.
        hvp_method (str, optional):
            - "autograd" - use pytorch autograd to calculate hessian-vector products.
            - "forward" - use two gradient evaluations to estimate hessian-vector products via froward finite differnce formula.
            - "central" - uses three gradient evaluations to estimate hessian-vector products via central finite differnce formula.
            Defaults to "autograd".
        h (float, optional): finite difference step size if :code:`hvp_method` is "forward" or "central". Defaults to 1e-3.
        warm_start (bool, optional):
            whether to warm-start conjugate gradient from previous solution. This can help if step size is small or maxiter is set to a small value. Defaults to False.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.

    Examples:
    .. code:: py
        ```
        # NewtonCG with backtracking line search
        opt = tz.Modular(model.parameters(), tz.m.NewtonCG(), tz.m.Backtracking())

        # Truncated newton
        opt = tz.Modular(model.parameters(), tz.m.NewtonCG(maxiter=10, warm_start=True), tz.m.Backtracking())

        # Newton preconditioning applied to momentum via CG
        # (in practice this is likely going to be unstable)
        opt = tz.Modular(
            model.parameters(),
            tz.m.NewtonCG(inner=tz.m.EMA(0.9)),
            tz.m.LR(0.1)
        )
        ```
    """
    def __init__(
        self,
        maxiter: int | None = None,
        tol: float = 1e-4,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        h: float = 1e-3,
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


