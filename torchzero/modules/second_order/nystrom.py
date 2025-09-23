from typing import Literal

import torch

from ...core import Chainable, Module, step, HVPMethod
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import hvp_fd_central, hvp_fd_forward
from ...utils.linalg.solve import nystrom_pcg, nystrom_sketch_and_solve, nystrom_approximation
from ...utils.linalg.linear_operator import Eigendecomposition, ScaledIdentity

class NystromSketchAndSolve(Module):
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
    opt = tz.Modular(
        model.parameters(),
        tz.m.NystromSketchAndSolve(100),
        tz.m.Backtracking()
    )
    ```

    Trust region NystromSketchAndSolve

    ```py
    opt = tz.Modular(
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
        del defaults['self'], defaults['inner']
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def update(self, var):
        update_freq = self.defaults["update_freq"]
        step = self.global_state.get("step", 0)
        self.global_state["step"] = step + 1

        if step % update_freq == 0:
            params = TensorList(var.params)

            closure = var.closure
            if closure is None: raise RuntimeError('NewtonCG requires closure')

            rank = self.defaults['rank']
            hvp_method = self.defaults['hvp_method']
            h = self.defaults['h']

            seed = self.defaults['seed']
            generator = self.get_generator(params[0].device, seed=seed)

            # ---------------------- Hessian vector product function --------------------- #
            _, H_mv, H_mm = var.tensor_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)

            # ---------------------------------- sketch ---------------------------------- #
            ndim = sum(t.numel() for t in var.params)
            device = params[0].device
            dtype = params[0].dtype

            try:
                L, Q = nystrom_approximation(A_mv=H_mv, A_mm=H_mm, ndim=ndim, rank=rank,
                                            dtype=dtype, device=device, generator=generator)

                self.global_state["L"] = L
                self.global_state["Q"] = Q
            except torch.linalg.LinAlgError:
                pass

    def apply(self, var):
        grad = var.get_grad()
        reg = self.defaults['reg']

        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
        if 'inner' in self.children:
            b = step(self.children['inner'], b, params=var.params, grads=grad, var=var)

        # ----------------------------------- solve ---------------------------------- #
        if "L" not in self.global_state:
            var.update = None
            return var

        L = self.global_state["L"]
        Q = self.global_state["Q"]
        x = nystrom_sketch_and_solve(L=L, Q=Q, b=torch.cat([t.ravel() for t in b]), reg=reg)

        # -------------------------------- set update -------------------------------- #
        var.update = vec_to_tensors(x, reference=var.params)
        return var

    def get_H(self, var=...):
        if "L" not in self.global_state:
            return ScaledIdentity()

        L = self.global_state["L"]
        Q = self.global_state["Q"]
        return Eigendecomposition(L, Q)


class NystromPCG(Module):
    """Newton's method with a Nyström-preconditioned conjugate gradient solver.
    This tends to outperform NewtonCG but requires tuning sketch size.
    An adaptive version exists in https://arxiv.org/abs/2110.02820, I might implement it too at some point.

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
    opt = tz.Modular(
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
        sketch_size: int,
        maxiter=None,
        tol=1e-8,
        reg: float = 1e-6,
        hvp_method: HVPMethod = "batched_autograd",
        h=1e-3,
        inner: Chainable | None = None,
        seed: int | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults)


        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def apply(self, var):
        params = TensorList(var.params)

        closure = var.closure
        if closure is None: raise RuntimeError('NystromPCG requires closure')

        settings = self.settings[params[0]]
        sketch_size = settings['sketch_size']
        maxiter = settings['maxiter']
        tol = settings['tol']
        reg = settings['reg']
        hvp_method = settings['hvp_method']
        h = settings['h']


        seed = settings['seed']
        generator = None
        if seed is not None:
            if 'generator' not in self.global_state:
                self.global_state['generator'] = torch.Generator(params[0].device).manual_seed(seed)
            generator = self.global_state['generator']


        # ---------------------- Hessian vector product function --------------------- #
        _, H_mv, H_mm = var.tensor_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)
        grad = var.get_grad()

        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
        if 'inner' in self.children:
            b = step(self.children['inner'], b, params=params, grads=grad, var=var)

        # ------------------------------ sketch&n&solve ------------------------------ #
        x = nystrom_pcg(A_mv=H_mv, A_mm=H_mm, b=torch.cat([t.ravel() for t in b]), sketch_size=sketch_size, reg=reg, tol=tol, maxiter=maxiter, x0_=None, generator=generator)
        var.update = vec_to_tensors(x, reference=params)
        return var


