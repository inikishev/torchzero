import torch

from ...core import Chainable, Transform
from ...utils import TensorList, unpack_dicts, unpack_states, vec_to_tensors_
from ...linalg.linear_operator import Dense


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
        n_first_step_samples: int = 10,
        start_step: int = 10,
        beta: float | None = None,
        damping: float = 1e-4,
        h: float = 1e-2,
        seed=None,
        update_freq: int = 1,
        inner: Chainable | None = None,
    ):
        defaults = dict(n_samples=n_samples, h=h, beta=beta, damping=damping, seed=seed, start_step=start_step, n_first_step_samples=n_first_step_samples)
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
        if k == 0: n_samples = fs["n_first_step_samples"]
        H_hat = None

        # compute new approximation
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

        # add damping
        if fs["damping"] != 0:
            reg = torch.eye(H_hat.size(0), device=H_hat.device, dtype=H_hat.dtype).mul_(fs["damping"])
            H_hat += reg

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
        updates = objective.get_updates()

        H: torch.Tensor = self.global_state["H"]
        k = self.global_state["step"]
        if k < fs["start_step"]:
            # don't precondition yet
            # I guess we can try using trace to scale the update
            # because it will have horrible scaling otherwise
            torch._foreach_div_(updates, H.trace())
            return objective

        b = torch.cat([t.ravel() for t in updates])
        sol = torch.linalg.lstsq(H, b).solution # pylint:disable=not-callable

        vec_to_tensors_(sol, updates)
        return objective

    def get_H(self, objective=...):
        return Dense(self.global_state["H"])


