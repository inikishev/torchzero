from typing import Literal

import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import NumberList, TensorList, unpack_states, unpack_dicts
from ..opt_utils import initial_step_size


class MatrixMomentum(Transform):
    """Second order momentum method.

    Matrix momentum is useful for convex objectives, also for some reason it has very really good generalization on elastic net logistic regression.

    Notes:
        - ``mu`` needs to be tuned very carefully. It is supposed to be smaller than (1/largest eigenvalue), otherwise this will be very unstable. I have devised an adaptive version of this - ``tz.m.AdaptiveMatrixMomentum``, and it works well without having to tune ``mu``, however the adaptive version doesn't work on stochastic objectives.

        - In most cases ``MatrixMomentum`` should be the first module in the chain because it relies on autograd.

        - This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument.

    Args:
        mu (float, optional): this has a similar role to (1 - beta) in normal momentum. Defaults to 0.1.
        hvp_method (str, optional):
            Determines how hessian-vector products are computed.

            - ``"batched_autograd"`` - uses autograd with batched hessian-vector products. If a single hessian-vector is evaluated, equivalent to ``"autograd"``. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd hessian-vector products. If multiple hessian-vector products are evaluated, uses a for-loop. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

            Defaults to ``"autograd"``.
        h (float, optional):
            The step size for finite difference if ``hvp_method`` is
            ``"fd_forward"`` or ``"fd_central"``. Defaults to 1e-3.
        hvp_tfm (Chainable | None, optional): optional module applied to hessian-vector products. Defaults to None.

    Reference:
        Orr, Genevieve, and Todd Leen. "Using curvature information for fast stochastic search." Advances in neural information processing systems 9 (1996).
    """

    def __init__(
        self,
        lr:float,
        mu=0.1,
        hvp_method: HVPMethod = "autograd",
        h: float = 1e-3,
        adaptive:bool = False,
        adapt_freq: int | None = None,

        inner: Chainable | None = None,
    ):
        defaults = dict(lr=lr, mu=mu, hvp_method=hvp_method, h=h, adaptive=adaptive, adapt_freq=adapt_freq)
        super().__init__(defaults, inner=inner)

    def reset_for_online(self):
        super().reset_for_online()
        self.clear_state_keys('p_prev')

    @torch.no_grad
    def update_states(self, objective, states, settings):
        step = self.increment_counter("step", 0)
        p = TensorList(objective.params)
        p_prev = unpack_states(states, p, 'p_prev', init=p)

        fs = settings[0]
        hvp_method = fs['hvp_method']
        h = fs['h']

        if step > 0:
            s = p - p_prev

            Hs, _ = objective.hessian_vector_product(s, at_x0=True, rgrad=None, hvp_method=hvp_method, h=h, retain_graph=False)
            Hs = [t.detach() for t in Hs]

            self.store(p, ("Hs", "s"), (Hs, s))

            # -------------------------------- adaptive mu ------------------------------- #
            if fs["adaptive"]:
                g = TensorList(objective.get_grads())

                if fs["adapt_freq"] is None:
                    # ---------------------------- deterministic case ---------------------------- #
                    g_prev = unpack_states(states, p, "g_prev", cls=TensorList)
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
                        assert objective.closure is not None
                        params = TensorList(objective.params)
                        p_cur = params.clone()

                        # move to previous params and evaluate p_prev with current mini-batch
                        params.copy_(unpack_states(states, p, 'p_prev'))
                        with torch.enable_grad():
                            objective.closure()
                        g_prev = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                        y = g - g_prev

                        # move back to current params
                        params.copy_(p_cur)

                        denom = y.global_vector_norm()
                        denom = denom.clip(min=torch.finfo(denom.dtype).tiny * 2)
                        self.global_state["mu_mul"] = s.global_vector_norm() / denom

        torch._foreach_copy_(p_prev, objective.params)

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        update = TensorList(objective.get_updates())
        lr, mu = unpack_dicts(settings, "lr", 'mu', cls=NumberList)

        if "mu_mul" in self.global_state:
            mu = mu * self.global_state["mu_mul"]

        # --------------------------------- 1st step --------------------------------- #
        # p_prev is not available so make a small step
        step = self.global_state["step"]
        if step == 1:
            if self.defaults["adaptive"]:
                # initialize
                unpack_states(states, objective.params, "g_prev", init=objective.get_grads())

            update.mul_(lr) # separate so that initial_step_size can clip correctly
            update.mul_(initial_step_size(update, 1e-7))
            return objective

        # -------------------------- matrix momentum update -------------------------- #
        s, Hs = unpack_states(states, objective.params, 's', 'Hs', cls=TensorList)

        update.mul_(lr).sub_(s).add_(Hs*mu)
        objective.updates = update
        return objective
