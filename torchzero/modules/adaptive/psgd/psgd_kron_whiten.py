# pylint:disable=not-callable
"""all functions are from https://github.com/lixilinx/psgd_torch/blob/master/psgd.py"""
import math
import warnings
from typing import Literal

import torch

from ....core import Chainable, TensorTransform
from ....utils import NumberList, TensorList
from .psgd import (
    init_kron,
    precond_grad_kron,
    update_precond_kron_whiten_eq,
    update_precond_kron_whiten_q0p5eq1p5,
    update_precond_kron_whiten_qep,
    update_precond_kron_whiten_qeq,
    update_precond_kron_whiten_quad,
    update_precond_kron_whiten_quad4p,
)

# matches
class KronWhiten(TensorTransform):
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

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults["inner"], defaults["self"]
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
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
            state["update_precond"] = update_precond_kron_whiten_quad4p
            state["precond_grad"] = lambda QL, exprs, G: exprs[0](*QL[0], G) # it's exprA(*Q, G)

        else:
            state["precond_grad"] = precond_grad_kron
            if dQ == "QEP":
                state["update_precond"] = update_precond_kron_whiten_qep
            elif dQ == "EQ":
                state["update_precond"] = update_precond_kron_whiten_eq
            elif dQ == "QEQ":
                state["update_precond"] = update_precond_kron_whiten_qeq
            elif dQ == "QUAD":
                state["update_precond"] = update_precond_kron_whiten_quad
            else:
                assert (dQ == "Q0.5EQ1.5") or (dQ == "Q0p5EQ1p5"), f"Invalid choice for dQ: '{dQ}'"
                state["update_precond"] = update_precond_kron_whiten_q0p5eq1p5

    @torch.no_grad
    def multi_tensor_update(self, tensors, params, grads, loss, states, settings):

        # initialize on the fly if not initialized
        if any(state["QLs_exprs"] is None for state in states):

            scale = max([torch.mean((torch.abs(g))**4) for g in tensors])
            scale = (scale + settings[0]["damping"]**4)**(-1/8)

            for param, state, setting in zip(params, states, settings):
                if state["QLs_exprs"] is None:
                    state["QLs_exprs"] = init_kron(
                        param.squeeze(),
                        Scale=scale,
                        max_size=setting["max_dim"],
                        max_skew=setting["max_skew"],
                        dQ=setting["dQ"],
                    )


        # update preconditioners
        # (could also try per-parameter probability)
        if torch.rand([]) < settings[0]["update_probability"]: # update Q
            for tensor, state, setting in zip(tensors, states, settings):
                state["update_precond"](
                    *state["QLs_exprs"],
                    tensor.squeeze(),
                    lr=setting["lr_preconditioner"],
                    betaL=setting["betaL"],
                    damping=setting["damping"],
                )

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):

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

        return pre_tensors