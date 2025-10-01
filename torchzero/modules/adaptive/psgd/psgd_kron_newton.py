# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings
from typing import Literal

import torch

from ....core import Chainable, HVPMethod, Transform
from ....utils import NumberList, TensorList, Distributions
from .psgd import (
    init_kron,
    precond_grad_kron,
    update_precond_kron_newton_eq,
    update_precond_kron_newton_q0p5eq1p5,
    update_precond_kron_newton_qep,
    update_precond_kron_newton_qeq,
    update_precond_kron_newton_quad,
    update_precond_kron_newton_quad4p,
)

# matches
class PSGDKronNewton(Transform):
    """Kron hessian preconditioner from Preconditioned Stochastic Gradient Descent (see https://github.com/lixilinx/psgd_torch)

    Args:
        max_dim (int, optional): dimensions with size larger than this use diagonal preconditioner. Defaults to 10_000.
        max_skew (float, optional):
            if memory used by full preconditioner (dim^2) is larger than total number of elements in a parameter times ``max_skew``, it uses a diagonal preconditioner. Defaults to 1.0.
        init_scale (float | None, optional):
            initial scale of the preconditioner. If None, determined based on a heuristic. Defaults to None.
        lr_preconditioner (float, optional): learning rate of the preconditioner. Defaults to 0.1.
        betaL (float, optional): EMA factor for the L-smoothness constant wrt Q. Defaults to 0.9.
        damping (float, optional): adds small noise to gradient when updating the preconditioner. Defaults to 1e-9.
        grad_clip_max_amp (float, optional): clips amplitude of the update. Defaults to float("inf").
        update_probability (float, optional): probability of updating preconditioner on each step. Defaults to 1.0.
        dQ (str, optional): geometry for preconditioner update. Defaults to "Q0.5EQ1.5".
        balance_probability (float, optional):
            probablility of balancing the dynamic ranges of the factors of Q to avoid over/under-flow on each step. Defaults to 0.01.
        hvp_method (HVPMethod, optional): how to compute hessian-vector products. Defaults to 'autograd'.
        h (float, optional):
            if ``hvp_method`` is ``"fd_central"`` or ``"fd_forward"``, controls finite difference step size.
            Defaults to 1e-3.
        distribution (Distributions, optional):
            distribution for random vectors for hessian-vector products. Defaults to 'normal'.
        inner (Chainable | None, optional): preconditioning will be applied to output of this module. Defaults to None.


    ###Examples:

    Pure PSGD Kron Newton:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.KronNewton(),
        tz.m.LR(1e-3),
    )
    ```

    Applying preconditioner to momentum:
    ```py
    optimizer = tz.Optimizer(
        model.parameters(),
        tz.m.KronNewton(inner=tz.m.EMA(0.9)),
        tz.m.LR(1e-3),
    )
    ```
    """
    def __init__(
        self,
        max_dim: int = 10_000,
        max_skew: float = 1.0,
        init_scale: float | None = None,
        lr_preconditioner: float = 0.1,
        betaL: float = 0.9,
        damping: float = 1e-9,
        grad_clip_max_amp: float = float("inf"),
        update_probability: float= 1.0,
        dQ: Literal["QEP", "EQ", "QEQ", "QUAD",  "Q0.5EQ1.5", "Q0p5EQ1p5", "QUAD4P"] = "Q0.5EQ1.5",
        balance_probability: float = 0.01,

        hvp_method: HVPMethod = 'autograd',
        h: float = 1e-3,
        distribution: Distributions = 'normal',

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults["inner"], defaults["self"]
        super().__init__(defaults, inner=inner)


    def _initialize_state(self, param, state, setting):
        assert "initialized" not in state
        state["initialized"] = True

        # initialize preconditioners
        if setting["init_scale"] is None:
            warnings.warn("FYI: Will set the preconditioner initial scale on the fly. Recommend to set it manually.")
            state["QLs_exprs"] = None
        else:
            state["QLs_exprs"] = init_kron(
                param.squeeze(),
                Scale=setting["init_scale"],
                max_size=setting["max_dim"],
                max_skew=setting["max_skew"],
                dQ=setting["dQ"],
            )

        dQ = setting["dQ"]
        if dQ == "QUAD4P":
            assert torch.finfo(param.dtype).eps < 1e-6, "Directly fitting P needs at least single precision"
            state["update_precond"] = update_precond_kron_newton_quad4p
            state["precond_grad"] = lambda QL, exprs, G: exprs[0](*QL[0], G) # it's exprA(*Q, G)

        else:
            state["precond_grad"] = precond_grad_kron
            if dQ == "QEP":
                state["update_precond"] = update_precond_kron_newton_quad
            elif dQ == "EQ":
                state["update_precond"] = update_precond_kron_newton_qep
            elif dQ == "QEQ":
                state["update_precond"] = update_precond_kron_newton_eq
            elif dQ == "QUAD":
                state["update_precond"] = update_precond_kron_newton_qeq
            else:
                assert (dQ == "Q0.5EQ1.5") or (dQ == "Q0p5EQ1p5"), f"Invalid choice for dQ: '{dQ}'"
                state["update_precond"] = update_precond_kron_newton_q0p5eq1p5

    @torch.no_grad
    def update_states(self, objective, states, settings):

        # initialize states
        for param, state, setting in zip(objective.params, states, settings):
            if "initialized" not in state:
                self._initialize_state(param, state, setting)

        fs = settings[0]

        uninitialized = any(state["QLs_exprs"] is None for state in states)
        if (torch.rand([]) < fs["update_probability"]) or uninitialized:

            # hessian-vector product
            vs = TensorList(objective.params).sample_like(distribution=fs["distribution"])
            Hvs, _ = objective.hessian_vector_product(z=vs, rgrad=None, at_x0=True, hvp_method=fs["hvp_method"], h=fs["h"])

            # initialize on the fly (why does it use vs?)
            if uninitialized:

                scale = (sum([torch.sum(torch.abs(v)**2) for v in vs])/sum([v.numel() for v in vs])) ** (1/4) # (mean(|v|^2))^(1/4)

                scale = scale * (max([torch.mean((torch.abs(h))**4) for h in Hvs]) + fs["damping"]**4) ** (-1/8) # (mean(|v|^2))^(1/4) * (mean(|h|^4))^(-1/8)

                for h, state, setting in zip(Hvs, states, settings):
                    if state["QLs_exprs"] is None:
                        state["QLs_exprs"] = init_kron(
                            h.squeeze(),
                            Scale=scale,
                            max_size=setting["max_dim"],
                            max_skew=setting["max_skew"],
                            dQ=setting["dQ"],
                        )

            # update preconditioner
            for v, h, state, setting in zip(vs, Hvs, states, settings):
                state["update_precond"](
                    *state["QLs_exprs"],
                    v.squeeze(),
                    h.squeeze(),
                    lr=setting["lr_preconditioner"],
                    betaL=setting["betaL"],
                    damping=setting["damping"],
                    balance_prob=setting["balance_probability"]
                )

    @torch.no_grad
    def apply_states(self, objective, states, settings):

        params = objective.params
        tensors = objective.get_updates()
        pre_tensors = []

        # precondition
        for param, tensor, state in zip(params, tensors, states):
            t = state["precond_grad"](
                *state["QLs_exprs"],
                tensor.squeeze(),
            )
            pre_tensors.append(t.view_as(param))

        # norm clipping
        grad_clip_max_amp = settings[0]["grad_clip_max_amp"]
        if grad_clip_max_amp < math.inf:
            pre_tensors = TensorList(pre_tensors)
            num_params = sum(t.numel() for t in pre_tensors)

            avg_amp = pre_tensors.dot(pre_tensors.conj()).div(num_params).sqrt()

            if avg_amp > grad_clip_max_amp:
                torch._foreach_mul_(pre_tensors, grad_clip_max_amp / avg_amp)

        objective.updates = pre_tensors
        return objective