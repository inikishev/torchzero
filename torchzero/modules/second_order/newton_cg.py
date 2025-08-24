import warnings
import math
from typing import Literal, cast
from operator import itemgetter
import torch

from ...core import Chainable, Module, apply_transform
from ...utils import TensorList, as_tensorlist, tofloat
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward
from ...utils.linalg.solve import cg, minres, find_within_trust_radius
from ..trust_region.trust_region import default_radius

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
        tol: float = 1e-8,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        solver: Literal['cg', 'minres', 'minres_npc'] = 'cg',
        h: float = 1e-3,
        miniter:int = 1,
        warm_start=False,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

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

        self._num_hvps_last_step = 0
        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                self._num_hvps_last_step += 1
                with torch.enable_grad():
                    return TensorList(hvp(params, grad, x, retain_graph=True))

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    self._num_hvps_last_step += 1
                    return TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1])

            elif hvp_method == 'central':
                def H_mm(x):
                    self._num_hvps_last_step += 1
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
            d, _ = cg(A_mm=H_mm, b=b, x0=x0, tol=tol, maxiter=maxiter, miniter=self.defaults["miniter"],reg=reg)

        elif solver == 'minres':
            d = minres(A_mm=H_mm, b=b, x0=x0, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=False)

        elif solver == 'minres_npc':
            d = minres(A_mm=H_mm, b=b, x0=x0, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=True)

        else:
            raise ValueError(f"Unknown solver {solver}")

        if warm_start:
            assert x0 is not None
            x0.copy_(d)

        var.update = d

        self._num_hvps += self._num_hvps_last_step
        return var


class NewtonCGSteihaug(Module):
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
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        tol: float = 1e-8,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central"] = "forward",
        solver: Literal['cg', "minres"] = 'cg',
        h: float = 1e-3,
        max_attempts: int = 100,
        max_history: int = 100,
        boundary_tol: float = 1e-1,
        miniter: int = 1,
        rms_beta: float | None = None,
        adapt_tol: bool = True,
        npc_terminate: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        tol = self.defaults['tol'] * self.global_state.get('tol_mul', 1)
        solver = self.defaults['solver'].lower().strip()

        (reg, maxiter, hvp_method, h, max_attempts, boundary_tol,
         eta, nplus, nminus, rho_good, rho_bad, init, npc_terminate,
         miniter, max_history, adapt_tol) = itemgetter(
             "reg", "maxiter", "hvp_method", "h", "max_attempts", "boundary_tol",
             "eta", "nplus", "nminus", "rho_good", "rho_bad", "init", "npc_terminate",
             "miniter", "max_history", "adapt_tol",
        )(self.defaults)

        self._num_hvps_last_step = 0

        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                self._num_hvps_last_step += 1
                with torch.enable_grad():
                    return TensorList(hvp(params, grad, x, retain_graph=True))

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    self._num_hvps_last_step += 1
                    return TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1])

            elif hvp_method == 'central':
                def H_mm(x):
                    self._num_hvps_last_step += 1
                    return TensorList(hvp_fd_central(closure, params, x, h=h, normalize=True)[1])

            else:
                raise ValueError(hvp_method)


        # ------------------------- update RMS preconditioner ------------------------ #
        b = var.get_update()
        P_mm = None
        rms_beta = self.defaults["rms_beta"]
        if rms_beta is not None:
            exp_avg_sq = self.get_state(params, "exp_avg_sq", init=b, cls=TensorList)
            exp_avg_sq.mul_(rms_beta).addcmul(b, b, value=1-rms_beta)
            exp_avg_sq_sqrt = exp_avg_sq.sqrt().add_(1e-8)
            def _P_mm(x):
                return x / exp_avg_sq_sqrt
            P_mm = _P_mm

        # -------------------------------- inner step -------------------------------- #
        if 'inner' in self.children:
            b = apply_transform(self.children['inner'], b, params=params, grads=grad, var=var)
        b = as_tensorlist(b)

        # ------------------------------- trust region ------------------------------- #
        success = False
        d = None
        x0 = [p.clone() for p in params]
        solution = None

        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_radius = self.global_state.get('trust_radius', init)

            # -------------- make sure trust radius isn't too small or large ------------- #
            finfo = torch.finfo(x0[0].dtype)
            if trust_radius < finfo.tiny * 2:
                trust_radius = self.global_state['trust_radius'] = init
                if adapt_tol:
                    self.global_state["tol_mul"] = self.global_state.get("tol_mul", 1) * 0.1

            elif trust_radius > finfo.max / 2:
                trust_radius = self.global_state['trust_radius'] = init

            # ----------------------------------- solve ---------------------------------- #
            d = None
            if solution is not None and solution.history is not None:
                d = find_within_trust_radius(solution.history, trust_radius)

            if d is None:
                if solver == 'cg':
                    d, solution = cg(
                        A_mm=H_mm,
                        b=b,
                        tol=tol,
                        maxiter=maxiter,
                        reg=reg,
                        trust_radius=trust_radius,
                        miniter=miniter,
                        npc_terminate=npc_terminate,
                        history_size=max_history,
                        P_mm=P_mm,
                    )

                elif solver == 'minres':
                    d = minres(A_mm=H_mm, b=b, trust_radius=trust_radius, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=npc_terminate)

                else:
                    raise ValueError(f"unknown solver {solver}")

            # ---------------------------- update trust radius --------------------------- #
            self.global_state["trust_radius"], success = default_radius(
                params=params,
                closure=closure,
                f=tofloat(var.get_loss(False)),
                g=b,
                H=H_mm,
                d=d,
                trust_radius=trust_radius,
                eta=eta,
                nplus=nplus,
                nminus=nminus,
                rho_good=rho_good,
                rho_bad=rho_bad,
                boundary_tol=boundary_tol,

                init=init, # init isn't used because check_overflow=False
                state=self.global_state, # not used
                settings=self.defaults, # not used
                check_overflow=False, # this is checked manually to adapt tolerance
            )

        # --------------------------- assign new direction --------------------------- #
        assert d is not None
        if success:
            var.update = d

        else:
            var.update = params.zeros_like()

        self._num_hvps += self._num_hvps_last_step
        return var