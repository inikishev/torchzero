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

    Notes:
        * In most cases NewtonCGSteihaug should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply Newton preconditioning to another module's output.

        * This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

    Warning:
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

    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.NewtonCG(),
        tz.m.Backtracking()
    )
    ```

    Truncated Newton method (useful for large-scale problems):
    ```
    opt = tz.Modular(
        model.parameters(),
        tz.m.NewtonCG(maxiter=10),
        tz.m.Backtracking()
    )
    ```

    """
    def __init__(
        self,
        maxiter: int | None = None,
        tol: float = 1e-8,
        reg: float = 1e-8,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        solver: Literal['cg', 'minres', 'minres_npc'] = 'cg',
        h: float = 1e-3, # tuned 1e-4 or 1e-3
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
    """Newton's method with trust region and a matrix-free Steihaug-Toint conjugate gradient solver.

    Notes:
        * In most cases NewtonCGSteihaug should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply Newton preconditioning to another module's output.

        * This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted. Defaults to 0.0.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        rho_good (float, optional):
            if ratio of actual to predicted rediction is larger than this, trust region size is multiplied by `nplus`.
        rho_bad (float, optional):
            if ratio of actual to predicted rediction is less than this, trust region size is multiplied by `nminus`.
        init (float, optional): Initial trust region value. Defaults to 1.
        max_attempts (max_attempts, optional):
            maximum number of trust radius reductions per step. A zero update vector is returned when
            this limit is exceeded. Defaults to 10.
        max_history (int, optional):
            CG will store this many intermediate solutions, reusing them when trust radius is reduced
            instead of re-running CG. Each solution storage requires 2N memory. Defaults to 100.
        boundary_tol (float | None, optional):
            The trust region only increases when suggested step's norm is at least `(1-boundary_tol)*trust_region`.
            This prevents increasing trust region when solution is not on the boundary. Defaults to 1e-2.

        maxiter (int | None, optional):
            maximum number of CG iterations per step. Each iteration requies one backward pass if `hvp_method="forward"`, two otherwise. Defaults to None.
        miniter (int, optional):
            minimal number of CG iterations. This prevents making no progress
        tol (float, optional):
            terminates CG when norm of the residual is less than this value. Defaults to 1e-8.
            when initial guess is below tolerance. Defaults to 1.
        reg (float, optional): hessian regularization. Defaults to 1e-8.
        solver (str, optional): solver, "cg" or "minres". "cg" is recommended. Defaults to 'cg'.
        adapt_tol (bool, optional):
            if True, whenever trust radius collapses to smallest representable number,
            the tolerance is multiplied by 0.1. Defaults to True.
        npc_terminate (bool, optional):
            whether to terminate CG/MINRES whenever negative curvature is detected. Defaults to False.

        hvp_method (str, optional):
            either "forward" to use forward formula which requires one backward pass per Hvp, or "central" to use a more accurate central formula which requires two backward passes. "forward" is usually accurate enough. Defaults to "forward".
        h (float, optional): finite difference step size. Defaults to 1e-3.

        inner (Chainable | None, optional):
            applies preconditioning to output of this module. Defaults to None.

    ### Examples:
    Trust-region Newton-CG:

    ```python
    opt = tz.Modular(
        model.parameters(),
        tz.m.NewtonCGSteihaug(),
    )
    ```

    ### Reference:
        Steihaug, Trond. "The conjugate gradient method and trust regions in large scale optimization." SIAM Journal on Numerical Analysis 20.3 (1983): 626-637.
    """
    def __init__(
        self,
        # trust region settings
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        max_attempts: int = 100,
        max_history: int = 100,
        boundary_tol: float = 1e-6, # tuned

        # cg settings
        maxiter: int | None = None,
        miniter: int = 1,
        tol: float = 1e-8,
        reg: float = 1e-8,
        solver: Literal['cg', "minres"] = 'cg',
        adapt_tol: bool = True,
        npc_terminate: bool = False,

        # hvp settings
        hvp_method: Literal["forward", "central"] = "central",
        h: float = 1e-3, # tuned 1e-4 or 1e-3

        # inner
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


        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
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