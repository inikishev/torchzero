from typing import Literal, overload
import torch

from ...utils import TensorList, as_tensorlist, NumberList
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply_transform, Module
from ...utils.linalg.solve import cg, steihaug_toint_cg, minres

class NewtonCG(Module):
    """Newton's method with a matrix-free conjugate gradient or minimial-residual solver.

    This optimizer implements Newton's method using a matrix-free conjugate
    gradient (CG) or a minimal-residual (MINRES) solver to approximate the search direction. Instead of
    forming the full Hessian matrix, it only requires Hessian-vector products
    (HVPs). These can be calculated efficiently using automatic
    differentiation or approximated using finite differences.

    .. note::
        In most cases NewtonCG should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

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


    """
    def __init__(
        self,
        maxiter: int | None = None,
        tol: float = 1e-4,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        solver: Literal['cg', 'minres', 'minres_npc'] = 'cg',
        h: float = 1e-3,
        warm_start=False,
        inner: Chainable | None = None,
    ):
        defaults = dict(tol=tol, maxiter=maxiter, reg=reg, hvp_method=hvp_method, solver=solver, h=h, warm_start=warm_start)
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
        solver = settings['solver'].lower().strip()
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
            b = apply_transform(self.children['inner'], b, params=params, grads=grad, var=var)
        b = as_tensorlist(b)

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state(params, 'prev_x', cls=TensorList) # initialized to 0 which is default anyway

        if solver == 'cg':
            x = cg(A_mm=H_mm, b=b, x0_=x0, tol=tol, maxiter=maxiter, reg=reg)

        elif solver == 'minres':
            x = minres(A_mm=H_mm, b=b, x0=x0, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=False)

        elif solver == 'minres_npc':
            x = minres(A_mm=H_mm, b=b, x0=x0, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=True)

        else:
            raise ValueError(f"Unknown solver {solver}")

        if warm_start:
            assert x0 is not None
            x0.copy_(x)

        var.update = x
        return var


class TruncatedNewtonCG(Module):
    """Trust region Newton's method with a matrix-free Steihaug-Toint conjugate gradient or MINRES solver.

    This optimizer implements Newton's method using a matrix-free conjugate
    gradient (CG) solver to approximate the search direction. Instead of
    forming the full Hessian matrix, it only requires Hessian-vector products
    (HVPs). These can be calculated efficiently using automatic
    differentiation or approximated using finite differences.

    .. note::
        In most cases NewtonCGSteihaug should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

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
        eta (float, optional):
            whenever actual to predicted loss reduction ratio is larger than this, a step is accepted.
        nplus (float, optional):
            trust region multiplier on successful steps.
        nminus (float, optional):
            trust region multiplier on unsuccessful steps.
        init (float, optional): initial trust region.
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
        inner (Chainable | None, optional):
            NewtonCG will attempt to apply preconditioning to the output of this module.

    Examples:
        Trust-region Newton-CG:

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.NewtonCGSteihaug(),
            )

    Reference:
        Steihaug, Trond. "The conjugate gradient method and trust regions in large scale optimization." SIAM Journal on Numerical Analysis 20.3 (1983): 626-637.
    """
    def __init__(
        self,
        maxiter: int | None = None,
        eta: float= 1e-6,
        nplus: float = 2,
        nminus: float = 0.25,
        init: float = 1,
        tol: float = 1e-4,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        solver: Literal['cg', 'minres', 'minres_npc'] = 'cg',
        h: float = 1e-3,
        max_attempts: int = 10,
        inner: Chainable | None = None,
    ):
        defaults = dict(tol=tol, maxiter=maxiter, reg=reg, hvp_method=hvp_method, h=h, eta=eta, nplus=nplus, nminus=nminus, init=init, max_attempts=max_attempts, solver=solver)
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
        max_attempts = settings['max_attempts']
        solver = settings['solver'].lower().strip()

        eta = settings['eta']
        nplus = settings['nplus']
        nminus = settings['nminus']
        init = settings['init']

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
            b = apply_transform(self.children['inner'], b, params=params, grads=grad, var=var)
        b = as_tensorlist(b)

        # ---------------------------------- run cg ---------------------------------- #
        success = False
        x = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', init)
            if trust_region < 1e-8 or trust_region > 1e8:
                trust_region = self.global_state['trust_region'] = init

            if solver == 'cg':
                x = steihaug_toint_cg(A_mm=H_mm, b=b, trust_region=trust_region, tol=tol, maxiter=maxiter, reg=reg)

            elif solver == 'minres':
                x = minres(A_mm=H_mm, b=b, trust_region=trust_region, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=False)

            elif solver == 'minres_npc':
                x = minres(A_mm=H_mm, b=b, trust_region=trust_region, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=True)

            else:
                raise ValueError(f"unknown solver {solver}")

            # ------------------------------- trust region ------------------------------- #
            Hx = H_mm(x)
            pred_reduction = b.dot(x) - 0.5 * x.dot(Hx)

            params -= x
            loss_star = closure(False)
            params += x
            reduction = var.get_loss(False) - loss_star

            rho = reduction / (pred_reduction.clip(min=1e-8))

            # failed step
            if rho < 0.25:
                self.global_state['trust_region'] = trust_region * nminus

            # very good step
            elif rho > 0.75:
                diff = trust_region - x.abs()
                if (diff.global_min() / trust_region) > 1e-4: # hits boundary
                    self.global_state['trust_region'] = trust_region * nplus

            # if the ratio is high enough then accept the proposed step
            if rho > eta:
                success = True

        assert x is not None
        if success:
            var.update = x

        else:
            var.update = params.zeros_like()

        return var


