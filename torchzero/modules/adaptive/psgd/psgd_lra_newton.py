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
class LRANewton(Transform):
    def __init__(
        self,
        rank: int = 10,
        init_scale: float | None = None,
        lr_preconditioner=0.1,
        betaL=0.9,
        damping=1e-9,
        momentum=0.0,
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