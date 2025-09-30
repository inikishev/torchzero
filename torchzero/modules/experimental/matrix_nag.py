from collections.abc import Callable
from typing import Literal

import torch
from torchzero.core import Chainable, Transform, HVPMethod
from torchzero.utils import NumberList, TensorList


def matrix_nag_(
    tensors_: TensorList,
    s: TensorList,
    Hvp_fn: Callable,
    mu: float | NumberList,
):
    s += tensors_
    Hv = TensorList(Hvp_fn(s))
    s -= Hv.mul_(mu)
    return tensors_.add_(s)


class MatrixNAG(Transform):
    """nesterov momentum version of matrix momentum. It seemed to work really well but adapting doesn't work,
    I need to test more"""
    def __init__(
        self,
        mu=0.1,
        hvp_method: HVPMethod = "autograd",
        h: float = 1e-3,
        adaptive:bool = False,
        adapt_freq: int | None = None,
        hvp_tfm: Chainable | None = None,
    ):
        defaults = dict(mu=mu, hvp_method=hvp_method, h=h, adaptive=adaptive, adapt_freq=adapt_freq)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('p_prev')

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        assert objective.closure is not None
        step = self.global_state.get("step", 0)
        self.global_state["step"] = step + 1

        p = TensorList(objective.params)
        g = TensorList(objective.get_grads(create_graph=self.defaults["hvp_method"] == "autograd"))
        p_prev = self.get_state(p, "p_prev", init=p, cls=TensorList)
        s = p - p_prev
        p_prev.copy_(p)

        # -------------------------------- adaptive mu ------------------------------- #
        if self.defaults["adaptive"]:

            if step == 1:
                self.global_state["mu_mul"] = 0

            else:
                # ---------------------------- deterministic case ---------------------------- #
                if self.defaults["adapt_freq"] is None:
                    g_prev = self.get_state(objective.params, "g_prev", cls=TensorList)
                    y = g - g_prev
                    g_prev.copy_(g)

                    denom =  y.global_vector_norm()
                    denom = denom.clip(min = torch.finfo(denom.dtype).tiny * 2)
                    self.global_state["mu_mul"] = s.global_vector_norm() / denom

                # -------------------------------- stochastic -------------------------------- #
                else:
                    adapt_freq = self.defaults["adapt_freq"]

                    # we start on 1nd step, and want to adapt when we start, so use (step - 1)
                    if (step - 1) % adapt_freq == 0:
                        assert objective.closure is not None
                        p_cur = p.clone()

                        # move to previous params and evaluate p_prev with current mini-batch
                        p.copy_(self.get_state(objective.params, 'p_prev'))
                        with torch.enable_grad():
                            objective.closure()
                        g_prev = [t.grad if t.grad is not None else torch.zeros_like(t) for t in p]
                        y = g - g_prev

                        # move back to current params
                        p.copy_(p_cur)

                        denom =  y.global_vector_norm()
                        denom = denom.clip(min = torch.finfo(denom.dtype).tiny * 2)
                        self.global_state["mu_mul"] = s.global_vector_norm() / denom

        # -------------------------- matrix momentum update -------------------------- #
        mu = self.get_settings(p, "mu", cls=NumberList)
        if "mu_mul" in self.global_state:
            mu = mu * self.global_state["mu_mul"]

        # def Hvp_fn(v):
        #     Hv, _ = self.Hvp(
        #         v=v,
        #         at_x0=True,
        #         var=objective,
        #         rgrad=g,
        #         hvp_method=self.defaults["hvp_method"],
        #         h=self.defaults["h"],
        #         normalize=True,
        #         retain_grad=False,
        #     )
        #     return Hv

        _, Hvp_fn = objective.list_Hvp_function(hvp_method=self.defaults["hvp_method"], h=self.defaults["h"], at_x0=True)

        objective.updates = matrix_nag_(
            tensors_=TensorList(objective.get_updates()),
            s=s,
            Hvp_fn=Hvp_fn,
            mu=mu,
        )

        return objective