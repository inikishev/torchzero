from collections.abc import Callable
from typing import Literal

import torch

from ...core import Chainable, Transform, Objective, HessianMethod, Module
from ...utils import vec_to_tensors, vec_to_tensors_
from ...linalg.linear_operator import Dense, DenseWithInverse
from ...linalg import torch_linalg

def _lu_solve(H: torch.Tensor, g: torch.Tensor):
    try:
        x, info = torch_linalg.solve_ex(H, g, retry_float64=True)
        if info == 0: return x
        return None
    except RuntimeError:
        return None

def _cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    L, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        return torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    return None

def _least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable


def _eigh_solve(H: torch.Tensor, g: torch.Tensor, tfm: Callable | None, search_negative: bool):
    try:
        L, Q = torch_linalg.eigh(H, retry_float64=True)
        if tfm is not None: L = tfm(L)
        if search_negative and L[0] < 0:
            neg_mask = L < 0
            Q_neg = Q[:, neg_mask] * L[neg_mask]
            return (Q_neg * (g @ Q_neg).sign()).mean(1)

        return Q @ ((Q.mH @ g) / L)

    except torch.linalg.LinAlgError:
        return None

def _newton_step(objective: Objective, H: torch.Tensor, damping:float, H_tfm, eigval_fn, use_lstsq:bool, g_proj: Callable | None = None, no_inner: Module | None = None) -> torch.Tensor:
    """INNER SHOULD BE NONE IN MOST CASES! Because Transform already has inner.
    Returns the update tensor, then do vec_to_tensor(update, params)"""
    # -------------------------------- inner step -------------------------------- #
    if no_inner is not None:
        objective = no_inner.step(objective)

    g = torch.cat([t.ravel() for t in objective.get_updates()])
    if g_proj is not None: g = g_proj(g)

    # ----------------------------------- solve ---------------------------------- #
    dir = None

    if damping != 0:
        H = H + torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(damping)

    if H_tfm is not None:
        ret = H_tfm(H, g)

        if isinstance(ret, torch.Tensor):
            dir = ret

        else: # returns (H, is_inv)
            H, is_inv = ret
            if is_inv: dir = H @ g

    if eigval_fn is not None:
        dir = _eigh_solve(H, g, eigval_fn, search_negative=False)

    if dir is None and use_lstsq: dir = _least_squares_solve(H, g)
    if dir is None: dir = _cholesky_solve(H, g)
    if dir is None: dir = _lu_solve(H, g)
    if dir is None: dir = _least_squares_solve(H, g)

    return dir

def _get_H(H: torch.Tensor, eigval_fn):
    if eigval_fn is not None:
        try:
            L, Q = torch.linalg.eigh(H) # pylint:disable=not-callable
            L: torch.Tensor = eigval_fn(L)
            H = Q @ L.diag_embed() @ Q.mH
            H_inv = Q @ L.reciprocal().diag_embed() @ Q.mH
            return DenseWithInverse(H, H_inv)

        except torch.linalg.LinAlgError:
            pass

    return Dense(H)

class Newton(Transform):
    """Exact Newton's method via autograd.

    Newton's method produces a direction jumping to the stationary point of quadratic approximation of the target function.
    The update rule is given by ``(H + yI)⁻¹g``, where ``H`` is the hessian and ``g`` is the gradient, ``y`` is the ``damping`` parameter.

    ``g`` can be output of another module, if it is specifed in ``inner`` argument.

    Note:
        In most cases Newton should be the first module in the chain because it relies on autograd. Use the ``inner`` argument if you wish to apply Newton preconditioning to another module's output.

    Note:
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating the hessian.
        The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        damping (float, optional): tikhonov regularizer value. Defaults to 0.
        update_freq (int, optional):
            updates hessian every ``update_freq`` steps. If this value is large,
            consider using ``tz.m.Chord`` which also pre-computes hessian inverse. Defaults to 1.
        eigval_fn (Callable | None, optional):
            optional eigenvalues transform, for example ``torch.abs`` or ``lambda L: torch.clip(L, min=1e-8)``.
            If this is specified, eigendecomposition will be used to invert the hessian.
        H_tfm (Callable | None, optional):
            optional hessian transforms, takes in two arguments - `(hessian, gradient)`.

            must return either a tuple: `(hessian, is_inverted)` with transformed hessian and a boolean value
            which must be True if transform inverted the hessian and False otherwise.

            Or it returns a single tensor which is used as the update.

            Defaults to None.
        hessian_method (str):
            Determines how hessian is computed.

            - ``"batched_autograd"`` - uses autograd to compute ``ndim`` batched hessian-vector products. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd to compute ``ndim`` hessian-vector products using for loop. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"functional_revrev"`` - uses ``torch.autograd.functional`` with "reverse-over-reverse" strategy and a for-loop. This is generally equivalent to ``"autograd"``.
            - ``"functional_fwdrev"`` - uses ``torch.autograd.functional`` with vectorized "forward-over-reverse" strategy. Faster than ``"functional_fwdrev"`` but uses more memory (``"batched_autograd"`` seems to be faster)
            - ``"func"`` - uses ``torch.func.hessian`` which uses "forward-over-reverse" strategy. This method is the fastest and is recommended, however it is more restrictive and fails with some operators which is why it isn't the default.
            - ``"gfd_forward"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"gfd_central"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a more accurate central formula which requires two gradient evaluations per hessian-vector product.
            - ``"fd"`` - uses function values to estimate gradient and hessian via finite difference. This uses less evaluations than chaining ``"gfd_*"`` after ``tz.m.FDM``.

            Defaults to ``"batched_autograd"``.
        h (float, optional):
            finite difference step size for "fd_forward" and "fd_central".
        use_lstsq (bool, Optional):
            if True, least squares will be used to solve the linear system, this may generate reasonable directions
            when hessian is not invertible. If False, tries cholesky, if it fails tries LU, and then least squares.
            If ``eigval_fn`` is specified, eigendecomposition will always be used to solve the linear system and this
            argument will be ignored.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.

    # See also

    * ``tz.m.NewtonCG``: uses a matrix-free conjugate gradient solver and hessian-vector products,
    useful for large scale problems as it doesn't form the full hessian.
    * ``tz.m.NewtonCGSteihaug``: trust region version of ``tz.m.NewtonCG``.
    * ``tz.m.InverseFreeNewton``: an inverse-free variant of Newton's method.
    * ``tz.m.quasi_newton``: large collection of quasi-newton methods that estimate the hessian.

    # Notes

    ## Implementation details

    ``(H + yI)⁻¹g`` is calculated by solving the linear system ``(H + yI)x = g``.
    The linear system is solved via cholesky decomposition, if that fails, LU decomposition, and if that fails, least squares.
    Least squares can be forced by setting ``use_lstsq=True``, which may generate better search directions when linear system is overdetermined.

    Additionally, if ``eigval_fn`` is specified, eigendecomposition of the hessian is computed,
    ``eigval_fn`` is applied to the eigenvalues, and ``(H + yI)⁻¹`` is computed using the computed eigenvectors and transformed eigenvalues. This is more generally more computationally expensive,
    but not by much

    ## Handling non-convexity

    Standard Newton's method does not handle non-convexity well without some modifications.
    This is because it jumps to the stationary point, which may be the maxima of the quadratic approximation.

    A modification to handle non-convexity is to modify the eignevalues to be positive,
    for example by setting ``eigval_fn = lambda L: L.abs().clip(min=1e-4)``.

    # Examples:

    Newton's method with backtracking line search

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Newton(),
        tz.m.Backtracking()
    )
    ```

    Newton's method for non-convex optimization.

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Newton(eigval_fn = lambda L: L.abs().clip(min=1e-4)),
        tz.m.Backtracking()
    )
    ```

    Newton preconditioning applied to momentum

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Newton(inner=tz.m.EMA(0.9)),
        tz.m.LR(0.1)
    )
    ```

    Diagonal newton example. This will still evaluate the entire hessian so it isn't efficient,
    but if you wanted to see how diagonal newton behaves or compares to full newton, you can use this.

    ```py
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Newton(H_tfm = lambda H, g: g/H.diag()),
        tz.m.Backtracking()
    )
    ```

    """
    def __init__(
        self,
        damping: float = 0,
        update_freq: int = 1,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        use_lstsq: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['update_freq'], defaults["inner"]
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        _, _, self.global_state['H'] = objective.hessian(
            hessian_method=fs['hessian_method'],
            h=fs['h'],
            at_x0=True
        )

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        params = objective.params
        fs = settings[0]

        update = _newton_step(
            objective=objective,
            H = self.global_state["H"],
            damping = fs["damping"],
            H_tfm = fs["H_tfm"],
            eigval_fn = fs["eigval_fn"],
            use_lstsq = fs["use_lstsq"],
        )

        objective.updates = vec_to_tensors(update, params)
        return objective

    def get_H(self,objective=...):
        return _get_H(self.global_state["H"], self.defaults["eigval_fn"])


class Chord(Transform):
    """Chord method. This computes hessian and it's inverse at initial solution and then re-uses it. It is possible to re-compute hessian every ``update_freq`` steps if it is specified.

    ``Newton`` computes hessian every ``update_freq`` steps and solves a linear system ``Hx = g`` on each step.

    ``Chord`` computes hessian and hessian inverse every ``update_freq`` steps and computes ``H^-1 @ g`` on each step
    with pre-computed ``H^-1``.

    Therefore ``Chord`` may be more efficient if ``update_freq`` is None or a large number.
    If ``update_freq = 1``, this is equivalent to Newton's method, but less efficient.

    Args:
        damping (float, optional): tikhonov regularizer value. Defaults to 0.
        update_freq (int, optional):
            updates hessian and hessian inverse every ``update_freq`` steps.
            If None, hessian is calculated once at the beginning. Defaults to None.
        eigval_fn (Callable | None, optional):
            optional eigenvalues transform, for example ``torch.abs`` or ``lambda L: torch.clip(L, min=1e-8)``.
            If this is specified, eigendecomposition will be used to invert the hessian.
        hessian_method (str):
            Determines how hessian is computed.

            - ``"batched_autograd"`` - uses autograd to compute ``ndim`` batched hessian-vector products. Faster than ``"autograd"`` but uses more memory.
            - ``"autograd"`` - uses autograd to compute ``ndim`` hessian-vector products using for loop. Slower than ``"batched_autograd"`` but uses less memory.
            - ``"functional_revrev"`` - uses ``torch.autograd.functional`` with "reverse-over-reverse" strategy and a for-loop. This is generally equivalent to ``"autograd"``.
            - ``"functional_fwdrev"`` - uses ``torch.autograd.functional`` with vectorized "forward-over-reverse" strategy. Faster than ``"functional_fwdrev"`` but uses more memory (``"batched_autograd"`` seems to be faster)
            - ``"func"`` - uses ``torch.func.hessian`` which uses "forward-over-reverse" strategy. This method is the fastest and is recommended, however it is more restrictive and fails with some operators which is why it isn't the default.
            - ``"gfd_forward"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a less accurate forward formula which requires one extra gradient evaluation per hessian-vector product.
            - ``"gfd_central"`` - computes ``ndim`` hessian-vector products via gradient finite difference using a more accurate central formula which requires two gradient evaluations per hessian-vector product.
            - ``"fd"`` - uses function values to estimate gradient and hessian via finite difference. This uses less evaluations than chaining ``"gfd_*"`` after ``tz.m.FDM``.

            Defaults to ``"batched_autograd"``.
        h (float, optional):
            finite difference step size for "fd_forward" and "fd_central".
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.
    """
    def __init__(
        self,
        damping: float = 0,
        update_freq: int | None = None,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["inner"]
        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        step = self.increment_counter("step", 0)
        fs = settings[0]
        update_freq = fs["update_freq"]

        # update on first step, or if `update_freq` triggers
        if ("H" not in self.global_state) or ((update_freq is not None) and (step % update_freq == 0)):
            _, _, H = objective.hessian(
                hessian_method=fs['hessian_method'],
                h=fs['h'],
                at_x0=True
            )

            if fs["damping"] != 0:
                H += torch.eye(H.size(0), device=H.device, dtype=H.dtype).mul_(fs["damping"])

            if fs["eigval_fn"] is not None:
                eigval_fn = fs["eigval_fn"]
                L, Q = torch_linalg.eigh(H, retry_float64=True)
                L = eigval_fn(L)
                H = (Q * L.unsqueeze(-2)) @ Q.mH
                H_inv = (Q * L.reciprocal().unsqueeze(-2)) @ Q.mH

            else:
                H_inv, _ = torch_linalg.inv_ex(H, retry_float64=True)

            self.global_state["H"] = H
            self.global_state["H_inv"] = H_inv

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()
        g = torch.cat([t.ravel() for t in objective.get_updates()])
        dir = self.global_state["H_inv"] @ g
        vec_to_tensors_(dir, updates)
        return objective

    def get_H(self,objective=...):
        return DenseWithInverse(self.global_state["H"], self.global_state["H_inv"])
