import torch

from ...core import Chainable, Transform
from ...utils import TensorList, unpack_dicts, unpack_states, vec_to_tensors
from ..second_order.newton import _get_H, _newton_step


def sg2_(
    delta_g: torch.Tensor,
    cd: torch.Tensor,
) -> torch.Tensor:
    """cd is c * perturbation."""

    M = torch.outer(0.5 / cd, delta_g)
    H_hat = 0.5 * (M + M.T)

    return H_hat



class SG2(Transform):
    """second-order stochastic gradient

    2SPSA (second-order SPSA)
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SPSA(),
        tz.m.SG2(),
        tz.m.LR(1e-2),
    )
    ```

    SG2 with line search
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.SG2(),
        tz.m.Backtracking()
    )
    ```

    SG2 with trust region
    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LevenbergMarquardt(tz.m.SG2(beta=0.75. n_samples=4)),
    )
    ```

    """

    def __init__(
        self,
        n_samples: int = 1,
        h: float = 1e-2,
        beta: float | None = None,
        damping: float = 0,
        eigval_fn=None,
        use_lstsq: bool = True,
        seed=None,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(n_samples=n_samples, h=h, beta=beta, damping=damping, eigval_fn=eigval_fn, seed=seed, use_lstsq=use_lstsq)
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]
        k = self.increment_counter("step", 0)

        params = TensorList(objective.params)
        closure = objective.closure
        if closure is None:
            raise RuntimeError("closure is required for SG2")
        generator = self.get_generator(params[0].device, self.defaults["seed"])

        h = unpack_dicts(settings, "h")
        x_0 = params.clone()
        n_samples = fs["n_samples"]
        H_hat = None

        for i in range(n_samples):
            # generate perturbation
            cd = params.rademacher_like(generator=generator).mul_(h)

            # two sided hessian approximation
            params.add_(cd)
            closure()
            g_p = params.grad.fill_none_(params)

            params.copy_(x_0)
            params.sub_(cd)
            closure()
            g_n = params.grad.fill_none_(params)

            delta_g = g_p - g_n

            # restore params
            params.set_(x_0)

            # compute H hat
            H_i = sg2_(
                delta_g = delta_g.to_vec(),
                cd = cd.to_vec(),
            )

            if H_hat is None: H_hat = H_i
            else: H_hat += H_i

        assert H_hat is not None
        if n_samples > 1: H_hat /= n_samples

        # update H
        H = self.global_state.get("H", None)
        if H is None: H = H_hat
        else:
            beta = fs["beta"]
            if beta is None: beta = (k+1) / (k+2)
            H.lerp_(H_hat, 1-beta)

        self.global_state["H"] = H


    @torch.no_grad
    def apply_states(self, objective, states, settings):
        fs = settings[0]

        dir = _newton_step(
            objective=objective,
            H = self.global_state["H"],
            damping = fs["damping"],
            H_tfm=None,
            eigval_fn=fs["eigval_fn"],
            use_lstsq=fs["use_lstsq"],
            g_proj=None,
        )

        objective.updates = vec_to_tensors(dir, objective.params)
        return objective

    def get_H(self,objective=...):
        return _get_H(self.global_state["H"], self.defaults["eigval_fn"])




# # two sided
# # we have g via x + d, x - d
# # H via g(x + d), g(x - d)
# # 1 is x, x+2d
# # 2 is x, x-2d
# # 5 evals in total

# # one sided
# # g via x, x + d
# # 1 is x, x + d
# # 2 is x, x - d
# # 3 evals and can use two sided for g_0

# class SPSA2(Module):
#     """second-order SPSA

#     SPSA2 with line search
#     ```python
#     opt = tz.Optimizer(
#         model.parameters(),
#         tz.m.SPSA2(),
#         tz.m.Backtracking()
#     )
#     ```

#     SPSA2 with trust region
#     ```python
#     opt = tz.Optimizer(
#         model.parameters(),
#         tz.m.LevenbergMarquardt(tz.m.SPSA2()),
#     )
#     ```
#     """

#     def __init__(
#         self,
#         n_samples: int = 1,
#         h: float = 1e-2,
#         beta: float | None = None,
#         damping: float = 0,
#         eigval_fn=None,
#         use_lstsq: bool = True,
#         seed=None,
#         inner: Chainable | None = None,
#     ):
#         defaults = dict(n_samples=n_samples, h=h, beta=beta, damping=damping, eigval_fn=eigval_fn, seed=seed, use_lstsq=use_lstsq)
#         super().__init__(defaults)

#         if inner is not None: self.set_child('inner', inner)

#     @torch.no_grad
#     def update(self, objective):
#         k = self.global_state.get('step', 0) + 1
#         self.global_state["step"] = k

#         params = TensorList(objective.params)
#         closure = objective.closure
#         if closure is None:
#             raise RuntimeError("closure is required for SPSA2")

#         generator = self.get_generator(params[0].device, self.defaults["seed"])

#         h = self.get_settings(params, "h")
#         x_0 = params.clone()
#         n_samples = self.defaults["n_samples"]
#         H_hat = None
#         g_0 = None

#         for i in range(n_samples):
#             # perturbations for g and H
#             cd_g = params.rademacher_like(generator=generator).mul_(h)
#             cd_H = params.rademacher_like(generator=generator).mul_(h)

#             # evaluate 4 points
#             x_p = x_0 + cd_g
#             x_n = x_0 - cd_g

#             params.set_(x_p)
#             f_p = closure(False)
#             params.add_(cd_H)
#             f_pp = closure(False)

#             params.set_(x_n)
#             f_n = closure(False)
#             params.add_(cd_H)
#             f_np = closure(False)

#             g_p_vec = (f_pp - f_p) / cd_H
#             g_n_vec = (f_np - f_n) / cd_H
#             delta_g = g_p_vec - g_n_vec

#             # restore params
#             params.set_(x_0)

#             # compute grad
#             g_i = (f_p - f_n) / (2 * cd_g)
#             if g_0 is None: g_0 = g_i
#             else: g_0 += g_i

#             # compute H hat
#             H_i = sg2_(
#                 delta_g = delta_g.to_vec().div_(2.0),
#                 cd = cd_g.to_vec(), # The interval is measured by the original 'cd'
#             )
#             if H_hat is None: H_hat = H_i
#             else: H_hat += H_i

#         assert g_0 is not None and H_hat is not None
#         if n_samples > 1:
#             g_0 /= n_samples
#             H_hat /= n_samples

#         # set grad to approximated grad
#         objective.grads = g_0

#         # update H
#         H = self.global_state.get("H", None)
#         if H is None: H = H_hat
#         else:
#             beta = self.defaults["beta"]
#             if beta is None: beta = k / (k+1)
#             H.lerp_(H_hat, 1-beta)

#         self.global_state["H"] = H

#     @torch.no_grad
#     def apply(self, objective):
#         dir = _newton_step(
#             objective=objective,
#             H = self.global_state["H"],
#             damping = self.defaults["damping"],
#             inner = self.children.get("inner", None),
#             H_tfm=None,
#             eigval_fn=self.defaults["eigval_fn"],
#             use_lstsq=self.defaults["use_lstsq"],
#             g_proj=None,
#         )

#         objective.updates = vec_to_tensors(dir, objective.params)
#         return objective

#     def get_H(self,objective=...):
#         return _get_H(self.global_state["H"], self.defaults["eigval_fn"])
