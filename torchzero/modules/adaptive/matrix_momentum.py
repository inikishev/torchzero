from typing import Literal
from collections.abc import Callable
import torch

from ...core import Module, apply_transform, Chainable
from ...utils import NumberList, TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward
from ..functional import initial_step_size


class MatrixMomentum(Module):
    """Second order momentum method.

    Matrix momentum is useful for convex objectives, also for some reason it has very really good generalization on elastic net logistic regression.

    Notes:
        - ``mu`` needs to be tuned very carefully. It is supposed to be smaller than (1/largest eigenvalue), otherwise this will be very unstable. I have devised an adaptive version of this - ``tz.m.AdaptiveMatrixMomentum``, and it works well without having to tune ``mu``, however the adaptive version doesn't work on stochastic objectives.

        - In most cases ``MatrixMomentum`` should be the first module in the chain because it relies on autograd.

        - This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument.

    Args:
        mu (float, optional): this has a similar role to (1 - beta) in normal momentum. Defaults to 0.1.
        hvp_method (str, optional):
            Determines how Hessian-vector products are evaluated.

            - ``"autograd"``: Use PyTorch's autograd to calculate exact HVPs.
              This requires creating a graph for the gradient.
            - ``"forward"``: Use a forward finite difference formula to
              approximate the HVP. This requires one extra gradient evaluation.
            - ``"central"``: Use a central finite difference formula for a
              more accurate HVP approximation. This requires two extra
              gradient evaluations.
            Defaults to "autograd".
        h (float, optional): finite difference step size if hvp_method is set to finite difference. Defaults to 1e-3.
        hvp_tfm (Chainable | None, optional): optional module applied to hessian-vector products. Defaults to None.

    Reference:
        Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """

    def __init__(
        self,
        lr:float,
        mu=0.1,
        hvp_method: Literal["autograd", "forward", "central"] = "autograd",
        h: float = 1e-3,
        adaptive:bool = False,
        adapt_freq: int | None = None,
        hvp_tfm: Chainable | None = None,
    ):
        defaults = dict(lr=lr, mu=mu, hvp_method=hvp_method, h=h, adaptive=adaptive, adapt_freq=adapt_freq)
        super().__init__(defaults)

        if hvp_tfm is not None:
            self.set_child('hvp_tfm', hvp_tfm)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('p_prev')

    @torch.no_grad
    def update(self, var):
        assert var.closure is not None
        p = TensorList(var.params)
        p_prev = self.get_state(p, 'p_prev', init=var.params)

        hvp_method = self.defaults['hvp_method']
        h = self.defaults['h']
        step = self.global_state.get("step", 0)
        self.global_state["step"] = step + 1

        if step > 0:
            s = p - p_prev

            Hs, _ = self.Hvp(s, at_x0=True, var=var, rgrad=None, hvp_method=hvp_method, h=h, normalize=True, retain_grad=False)
            Hs = [t.detach() for t in Hs]

            if 'hvp_tfm' in self.children:
                Hs = TensorList(apply_transform(self.children['hvp_tfm'], Hs, params=p, grads=var.grad, var=var))

            self.store(p, ("Hs", "s"), (Hs, s))

            # -------------------------------- adaptive mu ------------------------------- #
            if self.defaults["adaptive"]:
                g = TensorList(var.get_grad())

                if self.defaults["adapt_freq"] is None:
                    # ---------------------------- deterministic case ---------------------------- #
                    g_prev = self.get_state(var.params, "g_prev", cls=TensorList)
                    y = g - g_prev
                    g_prev.copy_(g)
                    denom = y.global_vector_norm()
                    denom = denom.clip(min=torch.finfo(denom.dtype).tiny * 2)
                    self.global_state["mu_mul"] = s.global_vector_norm() / denom

                else:
                    # -------------------------------- stochastic -------------------------------- #
                    adapt_freq = self.defaults["adapt_freq"]

                    # we start on 1nd step, and want to adapt when we start, so use (step - 1)
                    if (step - 1) % adapt_freq == 0:
                        assert var.closure is not None
                        params = TensorList(var.params)
                        p_cur = params.clone()

                        # move to previous params and evaluate p_prev with current mini-batch
                        params.copy_(self.get_state(var.params, 'p_prev'))
                        with torch.enable_grad():
                            var.closure()
                        g_prev = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                        y = g - g_prev

                        # move back to current params
                        params.copy_(p_cur)

                        denom = y.global_vector_norm()
                        denom = denom.clip(min=torch.finfo(denom.dtype).tiny * 2)
                        self.global_state["mu_mul"] = s.global_vector_norm() / denom

        torch._foreach_copy_(p_prev, var.params)

    @torch.no_grad
    def apply(self, var):
        update = TensorList(var.get_update())
        lr,mu = self.get_settings(var.params, "lr", 'mu', cls=NumberList)

        if "mu_mul" in self.global_state:
            mu = mu * self.global_state["mu_mul"]

        # --------------------------------- 1st step --------------------------------- #
        # p_prev is not available so make a small step
        step = self.global_state["step"]
        if step == 1:
            if self.defaults["adaptive"]: self.get_state(var.params, "g_prev", init=var.get_grad())
            update.mul_(lr) # separate so that initial_step_size can clip correctly
            update.mul_(initial_step_size(update, 1e-7))
            return var

        # -------------------------- matrix momentum update -------------------------- #
        s, Hs = self.get_state(var.params, 's', 'Hs', cls=TensorList)

        update.mul_(lr).sub_(s).add_(Hs*mu)
        var.update = update
        return var
