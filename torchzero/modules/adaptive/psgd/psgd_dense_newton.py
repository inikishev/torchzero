# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings
from typing import Literal

import torch

from ....core import Chainable, HVPMethod, Transform
from ....utils import Distributions, TensorList, vec_to_tensors_
from ._psgd_utils import _initialize_lra_state_
from .psgd import (
    lift2single,
    update_precond_dense_eq,
    update_precond_dense_q0p5eq1p5,
    update_precond_dense_qep,
    update_precond_dense_qeq,
    update_precond_dense_quad,
    update_precond_dense_quad4p,
)

# matches
class PSGDDenseNewton(Transform):
    """Dense hessian preconditioner from Preconditioned Stochastic Gradient Descent (see https://github.com/lixilinx/psgd_torch)

    Args:
        init_scale (float | None, optional):
            initial scale of the preconditioner. If None, determined based on a heuristic. Defaults to None.
        lr_preconditioner (float, optional): learning rate of the preconditioner. Defaults to 0.1.
        betaL (float, optional): EMA factor for the L-smoothness constant wrt Q. Defaults to 0.9.
        damping (float, optional):
            adds small noise to hessian-vector product when updating the preconditioner. Defaults to 1e-9.
        grad_clip_max_norm (float, optional): clips norm of the update. Defaults to float("inf").
        update_probability (float, optional): probability of updating preconditioner on each step. Defaults to 1.0.
        dQ (str, optional): geometry for preconditioner update. Defaults to "Q0.5EQ1.5".
        hvp_method (HVPMethod, optional): how to compute hessian-vector products. Defaults to 'autograd'.
        h (float, optional):
            if ``hvp_method`` is ``"fd_central"`` or ``"fd_forward"``, controls finite difference step size.
            Defaults to 1e-3.
        distribution (Distributions, optional):
            distribution for random vectors for hessian-vector products. Defaults to 'normal'.

        inner (Chainable | None, optional): preconditioning will be applied to output of this module. Defaults to None.

    ###Examples:

    Pure Dense Newton PSGD:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.DenseNewton(),
        tz.m.LR(1e-3),
    )
    ```

    Applying preconditioner to momentum:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.DenseNewton(inner=tz.m.EMA(0.9)),
        tz.m.LR(1e-3),
    )
    ```
    """
    def __init__(
        self,
        init_scale: float | None = None,
        lr_preconditioner=0.1,
        betaL=0.9,
        damping=1e-9,
        grad_clip_max_norm=float("inf"),
        update_probability=1.0,
        dQ: Literal["QUAD4P", "QUAD", "QEP", "EQ", "QEQ", "Q0p5EQ1p5", "Q0.5EQ1.5"] = "Q0.5EQ1.5",

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

        # -------------------------------- initialize -------------------------------- #
        if "Q" not in self.global_state:

            p = objective.params[0]
            dQ = fs["dQ"]
            init_scale = fs["init_scale"]

            if init_scale is None:
                self.global_state["Q"] = None

            else:
                n = sum(p.numel() for p in objective.params)
                if dQ == "QUAD4P":
                    init_scale *= init_scale
                self.global_state["Q"] = torch.eye(n, dtype=p.dtype, device=p.device) * init_scale

            self.global_state["L"] = lift2single(torch.zeros([], dtype=p.dtype, device=p.device)) # Lipschitz smoothness constant estimation for the psgd criterion

            if dQ == "QUAD4P":
                self.global_state["update_precond"] = update_precond_dense_quad4p
                self.global_state["precond_grad"] = lambda Q, g: Q @ g
                assert torch.finfo(p.dtype).eps < 1e-6, "Directly fitting P needs at least single precision"

            elif dQ == "QUAD":
                self.global_state["update_precond"] = update_precond_dense_quad
                self.global_state["precond_grad"] = lambda Q, g: Q @ (Q @ g) # Q is symmetric; just save one transpose

            else:
                self.global_state["precond_grad"] = lambda Q, g: Q.T @ (Q @ g)
                if dQ == "QEP":
                    self.global_state["update_precond"] = update_precond_dense_qep
                elif dQ == "EQ":
                    self.global_state["update_precond"] = update_precond_dense_eq
                elif dQ == "QEQ":
                    self.global_state["update_precond"] = update_precond_dense_qeq
                else:
                    assert (dQ == "Q0p5EQ1p5") or (dQ == "Q0.5EQ1.5"), f"Invalid choice for dQ: '{dQ}'"
                    self.global_state["update_precond"] = update_precond_dense_q0p5eq1p5

        # ---------------------------------- update ---------------------------------- #
        Q = self.global_state["Q"]
        if (torch.rand([]) < fs["update_probability"]) or Q is None:

            # hessian-vector product
            vs = TensorList(objective.params).sample_like(distribution=fs["distribution"])
            Hvs, _ = objective.hessian_vector_product(z=vs, rgrad=None, at_x0=True, hvp_method=fs["hvp_method"], h=fs["h"])

            v = torch.cat([t.ravel() for t in vs]).unsqueeze(1)
            h = torch.cat([t.ravel() for t in Hvs]).unsqueeze(1)

            # initialize on the fly
            if Q is None:
                scale = (torch.mean(v*v))**(1/4) * (torch.mean(h**4) + fs["damping"]**4)**(-1/8)
                if fs["dQ"] == "QUAD4P": # Q actually is P in this case
                    scale *= scale
                Q = self.global_state["Q"] = torch.eye(len(v), dtype=v.dtype, device=v.device) * scale

            # update preconditioner
            self.global_state["update_precond"](
                Q=Q,
                L=self.global_state["L"],
                v=v,
                h=h,
                lr=fs["lr_preconditioner"],
                betaL=fs["betaL"],
                damping=fs["damping"],
            )

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()

        # cat grads
        g = torch.cat([t.ravel() for t in updates]).unsqueeze(1) # column vec
        pre_grad = self.global_state["precond_grad"](self.global_state["Q"], g)

        # norm clipping
        grad_clip_max_norm = settings[0]["grad_clip_max_norm"]
        if grad_clip_max_norm < float("inf"): # clip preconditioned gradient
            grad_norm = torch.linalg.vector_norm(pre_grad)
            if grad_norm > grad_clip_max_norm:
                pre_grad *= grad_clip_max_norm / grad_norm

        vec_to_tensors_(pre_grad, updates)
        return objective