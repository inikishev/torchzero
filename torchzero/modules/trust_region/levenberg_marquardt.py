# pylint:disable=not-callable
import torch

from ...core import Chainable, Module
from ...linalg import linear_operator
from .trust_region import _RADIUS_KEYS, TrustRegionBase, _RadiusStrategy


class LevenbergMarquardt(TrustRegionBase):
    """Levenberg-Marquardt trust region algorithm.


    Args:
        hess_module (Module | None, optional):
            A module that maintains a hessian approximation (not hessian inverse!).
            This includes all full-matrix quasi-newton methods, ``tz.m.Newton`` and ``tz.m.GaussNewton``.
            When using quasi-newton methods, set ``inverse=False`` when constructing them.
        y (float, optional):
            when ``y=0``, identity matrix is added to hessian, when ``y=1``, diagonal of the hessian approximation
            is added. Values between interpolate. This should only be used with Gauss-Newton. Defaults to 0.
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted.
            When ``hess_module`` is ``Newton`` or ``GaussNewton``, this can be set to 0. Defaults to 0.15.
        nplus (float, optional): increase factor on successful steps. Defaults to 1.5.
        nminus (float, optional): decrease factor on unsuccessful steps. Defaults to 0.75.
        rho_good (float, optional):
            if ratio of actual to predicted rediction is larger than this, trust region size is multiplied by `nplus`.
        rho_bad (float, optional):
            if ratio of actual to predicted rediction is less than this, trust region size is multiplied by `nminus`.
        init (float, optional): Initial trust region value. Defaults to 1.
        update_freq (int, optional): frequency of updating the hessian. Defaults to 1.
        max_attempts (max_attempts, optional):
            maximum number of trust region size size reductions per step. A zero update vector is returned when
            this limit is exceeded. Defaults to 10.
        adaptive (bool, optional):
            if True, trust radius is multiplied by square root of gradient norm.
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    ### Examples:

    Gauss-Newton with Levenberg-Marquardt trust-region

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LevenbergMarquardt(tz.m.GaussNewton()),
    )
    ```

    LM-SR1
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LevenbergMarquardt(tz.m.SR1(inverse=False)),
    )
    ```

    """
    def __init__(
        self,
        hess_module: Chainable,
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        max_attempts: int = 10,
        radius_strategy: _RadiusStrategy | _RADIUS_KEYS = 'default',
        y: float = 0,
        adaptive: bool = False,
        fallback: bool = False,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(y=y, fallback=fallback, adaptive=adaptive)
        super().__init__(
            defaults=defaults,
            hess_module=hess_module,
            eta=eta,
            nplus=nplus,
            nminus=nminus,
            rho_good=rho_good,
            rho_bad=rho_bad,
            init=init,
            max_attempts=max_attempts,
            radius_strategy=radius_strategy,
            update_freq=update_freq,
            inner=inner,

            boundary_tol=None,
            radius_fn=None,
        )

    def trust_solve(self, f, g, H, radius, params, closure, settings):
        y = settings['y']
        adaptive = settings["adaptive"]

        if isinstance(H, linear_operator.DenseInverse):
            if settings['fallback']:
                H = H.to_dense()
            else:
                raise RuntimeError(
                    f"{self.children['hess_module']} maintains a hessian inverse. "
                    "LevenbergMarquardt requires the hessian, not the inverse. "
                    "If that module is a quasi-newton module, pass `inverse=False` on initialization. "
                    "Or pass `fallback=True` to LevenbergMarquardt to allow inverting the hessian inverse, "
                    "however that can be inefficient and unstable."
                )

        reg = 1/radius
        if adaptive: reg = reg * torch.linalg.vector_norm(g).sqrt()

        if y == 0:
            return H.solve_plus_diag(g, reg) # pyright:ignore[reportAttributeAccessIssue]

        diag = H.diagonal()
        diag = torch.where(diag < torch.finfo(diag.dtype).tiny * 2, 1, diag)
        if y != 1: diag = (diag*y) + (1-y)
        return H.solve_plus_diag(g, diag*reg)


