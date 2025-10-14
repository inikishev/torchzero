
from operator import itemgetter
from typing import Literal, cast

import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import TensorList, tofloat, unpack_dicts, unpack_states
from ...linalg.solve import cg, find_within_trust_radius, minres
from ..trust_region.trust_region import default_radius


class NewtonCG(Transform):
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

            - ``"autograd"`` - uses autograd hessian-vector products. If multiple hessian-vector products are evaluated, uses a for-loop.
            - ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

            For NewtonCG ``"batched_autograd"`` is equivalent to ``"autograd"``. Defaults to ``"autograd"``.
        h (float, optional):
            The step size for finite difference if ``hvp_method`` is
            ``"fd_forward"`` or ``"fd_central"``. Defaults to 1e-3.
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
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.NewtonCG(),
        tz.m.Backtracking()
    )
    ```

    Truncated Newton method (useful for large-scale problems):
    ```
    opt = tz.Optimizer(
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
        hvp_method: HVPMethod = "autograd",
        solver: Literal['cg', 'minres'] = 'cg',
        npc_terminate: bool = False,
        h: float = 1e-3, # tuned 1e-4 or 1e-3
        miniter:int = 1,
        warm_start=False,
        warm_beta:float=0,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults, inner=inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        hvp_method = fs['hvp_method']
        h = fs['h']

        # ---------------------- Hessian vector product function --------------------- #
        _, H_mv = objective.list_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)
        objective.temp = H_mv

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        self._num_hvps_last_step = 0
        H_mv = objective.poptemp()

        fs = settings[0]
        tol = fs['tol']
        reg = fs['reg']
        maxiter = fs['maxiter']
        solver = fs['solver'].lower().strip()
        warm_start = fs['warm_start']
        npc_terminate = fs["npc_terminate"]

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start:
            x0 = unpack_states(states, objective.params, 'prev_x', cls=TensorList)

        b = TensorList(objective.get_updates())

        if solver == 'cg':
            d, _ = cg(A_mv=H_mv, b=b, x0=x0, tol=tol, maxiter=maxiter,
                      miniter=fs["miniter"], reg=reg, npc_terminate=npc_terminate)

        elif solver == 'minres':
            d = minres(A_mv=H_mv, b=b, x0=x0, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=npc_terminate)

        else:
            raise ValueError(f"Unknown solver {solver}")

        if warm_start:
            assert x0 is not None
            x0.lerp_(d, weight = 1-fs["warm_beta"])

        objective.updates = d
        self._num_hvps += self._num_hvps_last_step
        return objective


class NewtonCGSteihaug(Transform):
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
            either ``"fd_forward"`` to use forward formula which requires one backward pass per hessian-vector product, or ``"fd_central"`` to use a more accurate central formula which requires two backward passes. ``"fd_forward"`` is usually accurate enough. Defaults to ``"fd_forward"``.
        h (float, optional): finite difference step size. Defaults to 1e-3.

        inner (Chainable | None, optional):
            applies preconditioning to output of this module. Defaults to None.

    ### Examples:
    Trust-region Newton-CG:

    ```python
    opt = tz.Optimizer(
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
        adapt_tol: bool = False,
        terminate_on_tr: bool = True,
        npc_terminate: bool = False,

        # hvp settings
        hvp_method: Literal["fd_forward", "fd_central"] = "fd_central",
        h: float = 1e-3, # tuned 1e-4 or 1e-3

        # inner
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults, inner=inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0


    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        hvp_method = fs['hvp_method']
        h = fs['h']

        # ---------------------- Hessian vector product function --------------------- #
        _, H_mv = objective.list_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)
        objective.temp = H_mv

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        self._num_hvps_last_step = 0

        H_mv = objective.poptemp()
        params = TensorList(objective.params)
        fs = settings[0]

        tol = fs['tol'] * self.global_state.get('tol_mul', 1)
        solver = fs['solver'].lower().strip()

        reg=fs["reg"]
        maxiter=fs["maxiter"]
        max_attempts=fs["max_attempts"]
        init=fs["init"]
        npc_terminate=fs["npc_terminate"]
        miniter=fs["miniter"]
        max_history=fs["max_history"]


        # ------------------------------- trust region ------------------------------- #
        success = False
        d = None
        orig_params = [p.clone() for p in params]
        b = TensorList(objective.get_updates())
        solution = None
        closure = objective.closure
        assert closure is not None

        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_radius = self.global_state.get('trust_radius', init)

            # -------------- make sure trust radius isn't too small or large ------------- #
            finfo = torch.finfo(orig_params[0].dtype)
            if trust_radius < finfo.tiny * 2:
                trust_radius = self.global_state['trust_radius'] = init

                if fs["adapt_tol"]:
                    self.global_state["tol_mul"] = self.global_state.get("tol_mul", 1) * 0.1

                if fs["terminate_on_tr"]:
                    objective.should_terminate = True

            elif trust_radius > finfo.max / 2:
                trust_radius = self.global_state['trust_radius'] = init

            # ----------------------------------- solve ---------------------------------- #
            d = None
            if solution is not None and solution.history is not None:
                d = find_within_trust_radius(solution.history, trust_radius)

            if d is None:
                if solver == 'cg':
                    d, solution = cg(
                        A_mv=H_mv,
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
                    d = minres(A_mv=H_mv, b=b, trust_radius=trust_radius, tol=tol, maxiter=maxiter, reg=reg, npc_terminate=npc_terminate)

                else:
                    raise ValueError(f"unknown solver {solver}")

            # ---------------------------- update trust radius --------------------------- #
            self.global_state["trust_radius"], success = default_radius(
                params = params,
                closure = closure,
                f = tofloat(objective.get_loss(False)),
                g = b,
                H = H_mv,
                d = d,
                trust_radius = trust_radius,
                eta = fs["eta"],
                nplus = fs["nplus"],
                nminus = fs["nminus"],
                rho_good = fs["rho_good"],
                rho_bad = fs["rho_bad"],
                boundary_tol = fs["boundary_tol"],

                init = cast(int, None), # init isn't used because check_overflow=False
                state = cast(dict, None), # not used
                settings = cast(dict, None), # not used
                check_overflow = False, # this is checked manually to adapt tolerance
            )

        # --------------------------- assign new direction --------------------------- #
        assert d is not None
        if success:
            objective.updates = d

        else:
            objective.updates = params.zeros_like()

        self._num_hvps += self._num_hvps_last_step
        return objective