from typing import Literal

import torch

from ...core import Chainable, Transform, HVPMethod
from ...utils import TensorList, vec_to_tensors
from ...linalg import nystrom_pcg, nystrom_sketch_and_solve, nystrom_approximation, cg
from ...linalg.linear_operator import Eigendecomposition, ScaledIdentity

class NystromSketchAndSolve(Transform):
    """Newton's method with a Nyström sketch-and-solve solver.

    Notes:
        - This module requires the a closure passed to the optimizer step, as it needs to re-evaluate the loss and gradients for calculating HVPs. The closure must accept a ``backward`` argument (refer to documentation).

        - In most cases NystromSketchAndSolve should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply Newton preconditioning to another module's output.

        - If this is unstable, increase the ``reg`` parameter and tune the rank.

    Args:
        rank (int): size of the sketch, this many hessian-vector products will be evaluated per step.
        reg (float, optional): regularization parameter. Defaults to 1e-3.
        hvp_method (str, optional):
            Determines how Hessian-vector products are computed.

            - ``"batched_autograd"`` - uses autograd with batched hessian-vector products to compute the preconditioner. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd hessian-vector products, uses a for loop to compute the preconditioner. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

            Defaults to ``"autograd"``.
        h (float, optional):
            The step size for finite difference if ``hvp_method`` is
            ``"fd_forward"`` or ``"fd_central"``. Defaults to 1e-3.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.
        seed (int | None, optional): seed for random generator. Defaults to None.


    Examples:
    NystromSketchAndSolve with backtracking line search

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.NystromSketchAndSolve(100),
        tz.m.Backtracking()
    )
    ```

    Trust region NystromSketchAndSolve

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.LevenbergMarquadt(tz.m.NystromSketchAndSolve(100)),
    )
    ```

    References:
    - [Frangella, Z., Rathore, P., Zhao, S., & Udell, M. (2024). SketchySGD: Reliable Stochastic Optimization via Randomized Curvature Estimates. SIAM Journal on Mathematics of Data Science, 6(4), 1173-1204.](https://arxiv.org/pdf/2211.08597)
    - [Frangella, Z., Tropp, J. A., & Udell, M. (2023). Randomized nyström preconditioning. SIAM Journal on Matrix Analysis and Applications, 44(2), 718-752](https://arxiv.org/abs/2110.02820)

    """
    def __init__(
        self,
        rank: int,
        reg: float = 1e-3,
        hvp_method: HVPMethod = "batched_autograd",
        h: float = 1e-3,
        update_freq: int = 1,
        inner: Chainable | None = None,
        seed: int | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner'], defaults["update_freq"]
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        params = TensorList(objective.params)
        fs = settings[0]

        # ---------------------- Hessian vector product function --------------------- #
        hvp_method = fs['hvp_method']
        h = fs['h']
        _, H_mv, H_mm = objective.tensor_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)

        # ---------------------------------- sketch ---------------------------------- #
        ndim = sum(t.numel() for t in objective.params)
        device = params[0].device
        dtype = params[0].dtype

        generator = self.get_generator(params[0].device, seed=fs['seed'])
        try:
            L, Q = nystrom_approximation(A_mv=H_mv, A_mm=H_mm, ndim=ndim, rank=fs['rank'],
                                        dtype=dtype, device=device, generator=generator)

            self.global_state["L"] = L
            self.global_state["Q"] = Q
        except torch.linalg.LinAlgError:
            pass

    def apply_states(self, objective, states, settings):
        fs = settings[0]
        b = objective.get_updates()

        # ----------------------------------- solve ---------------------------------- #
        if "L" not in self.global_state:
            return objective

        L = self.global_state["L"]
        Q = self.global_state["Q"]
        x = nystrom_sketch_and_solve(L=L, Q=Q, b=torch.cat([t.ravel() for t in b]), reg=fs["reg"])

        # -------------------------------- set update -------------------------------- #
        objective.updates = vec_to_tensors(x, reference=objective.params)
        return objective

    def get_H(self, objective=...):
        if "L" not in self.global_state:
            return ScaledIdentity()

        L = self.global_state["L"]
        Q = self.global_state["Q"]
        return Eigendecomposition(L, Q)


class NystromPCG(Transform):
    """Newton's method with a Nyström-preconditioned conjugate gradient solver.

    Notes:
        - This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating HVPs.
        The closure must accept a ``backward`` argument (refer to documentation).

        - In most cases NystromPCG should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply Newton preconditioning to another module's output.

    Args:
        sketch_size (int):
            size of the sketch for preconditioning, this many hessian-vector products will be evaluated before
            running the conjugate gradient solver. Larger value improves the preconditioning and speeds up
            conjugate gradient.
        maxiter (int | None, optional):
            maximum number of iterations. By default this is set to the number of dimensions
            in the objective function, which is supposed to be enough for conjugate gradient
            to have guaranteed convergence. Setting this to a small value can still generate good enough directions.
            Defaults to None.
        tol (float, optional): relative tolerance for conjugate gradient solver. Defaults to 1e-4.
        reg (float, optional): regularization parameter. Defaults to 1e-8.
        hvp_method (str, optional):
            Determines how Hessian-vector products are computed.

            - ``"batched_autograd"`` - uses autograd with batched hessian-vector products to compute the preconditioner. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd hessian-vector products, uses a for loop to compute the preconditioner. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"fd_forward"`` - uses gradient finite difference approximation with a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"fd_central"`` - uses gradient finite difference approximation with a more accurate central formula which requires two gradient evaluations per hessian-vector product.

            Defaults to ``"autograd"``.
        h (float, optional):
            The step size for finite difference if ``hvp_method`` is
            ``"fd_forward"`` or ``"fd_central"``. Defaults to 1e-3.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.
        seed (int | None, optional): seed for random generator. Defaults to None.

    Examples:

    NystromPCG with backtracking line search

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.NystromPCG(10),
        tz.m.Backtracking()
    )
    ```

    Reference:
        Frangella, Z., Tropp, J. A., & Udell, M. (2023). Randomized nyström preconditioning. SIAM Journal on Matrix Analysis and Applications, 44(2), 718-752. https://arxiv.org/abs/2110.02820

    """
    def __init__(
        self,
        rank: int,
        maxiter=None,
        tol=1e-8,
        reg: float = 1e-6,
        update_freq: int = 1, # here update_freq is within update_states
        hvp_method: HVPMethod = "batched_autograd",
        h=1e-3,
        inner: Chainable | None = None,
        seed: int | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        # ---------------------- Hessian vector product function --------------------- #
        # this should run on every update_states
        hvp_method = fs['hvp_method']
        h = fs['h']
        _, H_mv, H_mm = objective.tensor_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)
        objective.temp = H_mv

        # --------------------------- update preconditioner -------------------------- #
        step = self.increment_counter("step", 0)
        update_freq = self.defaults["update_freq"]

        if step % update_freq == 0:

            rank = fs['rank']
            ndim = sum(t.numel() for t in objective.params)
            device = objective.params[0].device
            dtype = objective.params[0].dtype
            generator = self.get_generator(device, seed=fs['seed'])

            try:
                L, Q = nystrom_approximation(A_mv=None, A_mm=H_mm, ndim=ndim, rank=rank,
                                            dtype=dtype, device=device, generator=generator)

                self.global_state["L"] = L
                self.global_state["Q"] = Q
            except torch.linalg.LinAlgError:
                pass

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        b = objective.get_updates()
        H_mv = objective.poptemp()
        fs = self.settings[objective.params[0]]

        # ----------------------------------- solve ---------------------------------- #
        if "L" not in self.global_state:
            # fallback on cg
            sol = cg(A_mv=H_mv, b=TensorList(b), tol=fs["tol"], reg=fs["reg"], maxiter=fs["maxiter"])
            objective.updates = sol.x
            return objective

        L = self.global_state["L"]
        Q = self.global_state["Q"]
        x = nystrom_pcg(L=L, Q=Q, A_mv=H_mv, b=torch.cat([t.ravel() for t in b]),
                        reg=fs['reg'], tol=fs["tol"], maxiter=fs["maxiter"])

        # -------------------------------- set update -------------------------------- #
        objective.updates = vec_to_tensors(x, reference=objective.params)
        return objective
