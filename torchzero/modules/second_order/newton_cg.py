from collections.abc import Callable
from typing import Literal, overload
import warnings
import torch

from ...utils import TensorList, as_tensorlist, generic_zeros_like, generic_vector_norm, generic_numel
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply_transform, Module
from ...utils.linalg.solve import cg

class NewtonCG(Module):
    """Newton's method with a matrix-free conjugate gradient solver.

    This optimizer implements Newton's method using a matrix-free conjugate
    gradient (CG) solver to approximate the search direction. Instead of
    forming the full Hessian matrix, it only requires Hessian-vector products
    (HVPs). These can be calculated efficiently using automatic
    differentiation or approximated using finite differences.

    .. note::
        In most cases NewtonCG should be the first module in the chain because it relies on extra autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

    .. note::
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).

    .. warning::
        CG may fail if hessian is not positive-definite.

    Args:
        maxiter (int | None, optional):
            Maximum number of iterations for the conjugate gradient solver.
            By default, this is set to the number of dimensions in the
            objective function, which is the theoretical upper bound for CG
            convergence. Setting this to a smaller value (truncated Newton)
            can still generate good search directions. Defaults to None.
        tol (float, optional):
            Relative tolerance for the conjugate gradient solver to determine
            convergence. Defaults to 1e-4.
        reg (float, optional):
            Regularization parameter (damping) added to the Hessian diagonal.
            This helps ensure the system is positive-definite. Defaults to 1e-8.
        hvp_method (str, optional):
            Determines how Hessian-vector products are evaluated.

            - ``"autograd"``: Use PyTorch's autograd to calculate exact HVPs.
              This requires creating a graph for the gradient.
            - ``"forward"``: Use a forward finite difference formula to
              approximate the HVP. This requires one extra gradient evaluation.
            - ``"central"``: Use a central finite difference formula for a
              more accurate HVP approximation. This requires two extra
              gradient evaluations.
            Defaults to "autograd".
        h (float, optional):
            The step size for finite differences if :code:`hvp_method` is
            ``"forward"`` or ``"central"``. Defaults to 1e-3.
        warm_start (bool, optional):
            If ``True``, the conjugate gradient solver is initialized with the
            solution from the previous optimization step. This can accelerate
            convergence, especially in truncated Newton methods.
            Defaults to False.
        inner (Chainable | None, optional):
            NewtonCG will attempt to apply preconditioning to the output of this module.

    Examples:
        Newton-CG with a backtracking line search:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.NewtonCG(),
                tz.m.Backtracking()
            )

        Truncated Newton method (useful for large-scale problems):

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.NewtonCG(maxiter=10, warm_start=True),
                tz.m.Backtracking()
            )

        Newton preconditioning applied to momentum (may be unstable):

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.NewtonCG(inner=tz.m.EMA(0.9)),
                tz.m.LR(0.1)
            )

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


