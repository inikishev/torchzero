from functools import partial

import numpy as np
import torch
from scipy.optimize import lsq_linear

from ...core import Chainable, Module
from ...utils import TensorList, vec_to_tensors
from .trust_region import TrustRegionBase, _update_tr_radius


def _flatten_tensors(tensors: list[torch.Tensor]):
    return torch.cat([t.ravel() for t in tensors])

class InfinityNormTrustRegion(TrustRegionBase):
    """Trust region with L-infinity norm via ``scipy.optimize.lsq_linear``.

    Args:
        hess_module (Module | None, optional):
            A module that maintains a hessian approximation (not hessian inverse!).
            This includes all full-matrix quasi-newton methods, ``tz.m.Newton`` and ``tz.m.GaussNewton``.
            When using quasi-newton methods, set `inverse=False` when constructing them.
        eta (float, optional):
            if ratio of actual to predicted rediction is larger than this, step is accepted.
            When :code:`hess_module` is GaussNewton, this can be set to 0. Defaults to 0.15.
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
        boundary_tol (float | None, optional):
            The trust region only increases when suggested step's norm is at least `(1-boundary_tol)*trust_region`.
            This prevents increasing trust region when solution is not on the boundary. Defaults to 1e-2.
        tol (float | None, optional): tolerance for least squares solver.
        fallback (bool, optional):
            if ``True``, when ``hess_module`` maintains hessian inverse which can't be inverted efficiently, it will
            be inverted anyway. When ``False`` (default), a ``RuntimeError`` will be raised instead.
        inner (Chainable | None, optional): preconditioning is applied to output of thise module. Defaults to None.

    Examples:
        BFGS with infinity-norm trust region

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.InfinityNormTrustRegion(hess_module=tz.m.BFGS(inverse=False)),
            )
    """
    def __init__(
        self,
        hess_module: Module,
        eta: float= 0.0,
        nplus: float = 3.5,
        nminus: float = 0.25,
        rho_good: float = 0.99,
        rho_bad: float = 1e-4,
        init: float = 1,
        update_freq: int = 1,
        max_attempts: int = 10,
        boundary_tol: float | None = None,
        prefer_dense:bool=True,
        tol: float = 1e-10,
        inner: Chainable | None = None,
    ):
        defaults = dict(init=init, nplus=nplus, nminus=nminus, eta=eta, tol=tol, max_attempts=max_attempts,boundary_tol=boundary_tol, rho_bad=rho_bad, prefer_dense=prefer_dense, rho_good=rho_good)
        super().__init__(hess_module=hess_module, defaults=defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def trust_region_apply(self, var, tensors, H):
        assert H is not None

        params = TensorList(var.params)
        settings = self.settings[params[0]]
        g = _flatten_tensors(tensors)

        tol = settings['tol']
        max_attempts = settings['max_attempts']
        prefer_dense = settings['prefer_dense']

        loss = var.loss
        closure = var.closure
        if closure is None: raise RuntimeError("Trust region requires closure")
        if loss is None: loss = var.get_loss(False)

        success = False
        d = None
        while not success:
            max_attempts -= 1
            if max_attempts < 0: break

            trust_region = self.global_state.get('trust_region', settings['init'])

            if trust_region < 1e-10 or trust_region > 1e16:
                trust_region = self.global_state['trust_region'] = settings['init']

            if prefer_dense and H.is_dense():
                # convert to array if possible to avoid many conversions
                # between torch and numpy, plus it seems that it uses
                # a better solver
                A = H.to_tensor().numpy(force=True).astype(np.float64)
            else:
                # memory efficient linear operator (is this still faster on CUDA?)
                A = H.scipy_linop()

            try:
                d_np = lsq_linear(
                    A,
                    g.numpy(force=True).astype(np.float64),
                    tol=tol,
                    bounds=(-trust_region, trust_region),
                ).x
                d = torch.as_tensor(d_np, device=g.device, dtype=g.dtype)
            except np.linalg.LinAlgError:
                self.children['hess_module'].reset()
                d = g

            self.global_state['trust_region'], success = _update_tr_radius(
                params=params, closure=closure, d=d, f=loss, g=g, H=H,
                trust_region=trust_region, settings=settings,
                boundary_fn=partial(torch.linalg.vector_norm, ord=torch.inf) # pylint:disable=not-callable
            )

        assert d is not None
        if success: var.update = vec_to_tensors(d, params)
        else: var.update = params.zeros_like()

        return var
