# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings

import torch

from ....core import Chainable, HVPMethod, Transform
from ....utils import Distributions, TensorList, vec_to_tensors_
from .psgd import lift2single, precond_grad_lra, update_precond_lra_newton
from ._psgd_utils import _initialize_lra_state_

# matches
class PSGDLRANewton(Transform):
    """Low rank hessian preconditioner from Preconditioned Stochastic Gradient Descent (see https://github.com/lixilinx/psgd_torch)

    Args:
        rank (int, optional):
            Preconditioner has a diagonal part and a low rank part, whose rank is decided by this setting. Defaults to 10.
        init_scale (float | None, optional):
            initial scale of the preconditioner. If None, determined based on a heuristic. Defaults to None.
        lr_preconditioner (float, optional): learning rate of the preconditioner. Defaults to 0.1.
        betaL (float, optional): EMA factor for the L-smoothness constant wrt Q. Defaults to 0.9.
        damping (float, optional):
            adds small noise to hessian-vector product when updating the preconditioner. Defaults to 1e-9.
        grad_clip_max_norm (float, optional): clips norm of the update. Defaults to float("inf").
        update_probability (float, optional): probability of updating preconditioner on each step. Defaults to 1.0.
        hvp_method (HVPMethod, optional): how to compute hessian-vector products. Defaults to 'autograd'.
        h (float, optional):
            if ``hvp_method`` is ``"fd_central"`` or ``"fd_forward"``, controls finite difference step size.
            Defaults to 1e-3.
        distribution (Distributions, optional):
            distribution for random vectors for hessian-vector products. Defaults to 'normal'.

        inner (Chainable | None, optional): preconditioning will be applied to output of this module. Defaults to None.

    ###Examples:

    Pure LRA Newton PSGD:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LRANewton(),
        tz.m.LR(1e-3),
    )
    ```

    Applying preconditioner to momentum:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.LRANewton(inner=tz.m.EMA(0.9)),
        tz.m.LR(1e-3),
    )
    ```
    """
    def __init__(
        self,
        rank: int = 10,
        init_scale: float | None = None,
        lr_preconditioner=0.1,
        betaL=0.9,
        damping=1e-9,
        grad_clip_max_norm=float("inf"),
        update_probability=1.0,

        hvp_method: HVPMethod = 'autograd',
        h: float = 1e-3,
        distribution: Distributions = 'normal',

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults["inner"], defaults["self"]
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        # initialize
        if "UVd" not in self.global_state:
            p = torch.cat([t.ravel() for t in objective.params])
            _initialize_lra_state_(p, self.global_state, fs)

        UVd = self.global_state["UVd"]
        if (torch.rand([]) < fs["update_probability"]) or (UVd[2] is None):

            # hessian-vector product
            vs = TensorList(objective.params).sample_like(distribution=fs["distribution"])
            Hvs, _ = objective.hessian_vector_product(z=vs, rgrad=None, at_x0=True, hvp_method=fs["hvp_method"], h=fs["h"])

            v = torch.cat([t.ravel() for t in vs]).unsqueeze(1)
            h = torch.cat([t.ravel() for t in Hvs]).unsqueeze(1)

            if UVd[2] is None:
                UVd[2] = (torch.mean(v*v))**(1/4) * (torch.mean(h**4) + fs["damping"]**4)**(-1/8) * torch.ones_like(v)

            # update preconditioner
            update_precond_lra_newton(UVd=UVd, Luvd=self.global_state["Luvd"], v=v, h=h, lr=fs["lr_preconditioner"], betaL=fs["betaL"], damping=fs["damping"])


    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()

        g = torch.cat([t.ravel() for t in updates]).unsqueeze(1) # column vec
        pre_grad = precond_grad_lra(UVd=self.global_state["UVd"], g=g)

        # norm clipping
        grad_clip_max_norm = settings[0]["grad_clip_max_norm"]
        if grad_clip_max_norm < float("inf"): # clip preconditioned gradient
            grad_norm = torch.linalg.vector_norm(pre_grad)
            if grad_norm > grad_clip_max_norm:
                pre_grad *= grad_clip_max_norm / grad_norm

        vec_to_tensors_(pre_grad, updates)
        return objective