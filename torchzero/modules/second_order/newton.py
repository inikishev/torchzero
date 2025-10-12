from collections.abc import Callable
from typing import Any

import torch

from ...core import Chainable, Transform, Objective, HessianMethod
from ...utils import vec_to_tensors_
from ...linalg.linear_operator import Dense, DenseWithInverse, Eigendecomposition
from ...linalg import torch_linalg
from ...linalg.eigh import regularize_eigh

def _try_lu_solve(H: torch.Tensor, g: torch.Tensor):
    try:
        x, info = torch_linalg.solve_ex(H, g, retry_float64=True)
        if info == 0: return x
        return None
    except RuntimeError:
        return None

def _try_cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    L, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        return torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    return None

def _least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable

def _newton_update_state_(
    state: dict,
    H: torch.Tensor,
    damping: float,
    eigval_fn: Callable | None,
    eigv_tol: float | None,
    truncate: int | None,
    precompute_inverse: bool,
    use_lstsq: bool,
):
    """used in most hessian-based modules"""
    # add damping
    if damping != 0:
        reg = torch.eye(H.size(0), device=H.device, dtype=H.dtype).mul_(damping)
        H += reg

    # if any args require eigendecomp, we don't need H or H_inv, we store factors
    if any(i is not None for i in [eigval_fn, eigv_tol, truncate]):
        L, Q = torch_linalg.eigh(H, retry_float64=True)
        if eigval_fn is not None: L = eigval_fn(L)
        L, Q = regularize_eigh(L, Q, truncate=truncate, tol=eigv_tol)
        state["L"] = L
        state["Q"] = Q
        return

    # pre-compute inverse if requested
    # store H to as it is needed for trust regions
    state["H"] = H
    if precompute_inverse:
        if use_lstsq:
            H_inv = torch.linalg.pinv(H) # pylint:disable=not-callable
        else:
            H_inv, _ = torch_linalg.inv_ex(H)
        state["H_inv"] = H_inv


def _newton_solve(
    b: torch.Tensor,
    state: dict[str, torch.Tensor | Any],
    use_lstsq: bool = False,
):
    """
    used in most hessian-based modules. state is from ``_newton_update_state_``, in it:

    H (torch.Tensor): hessian
    H_inv (torch.Tensor | None): hessian inverse
    L (torch.Tensor | None): eigenvalues (transformed)
    Q (torch.Tensor | None): eigenvectors
    """
    # use eig if provided
    if "L" in state:
        Q = state["Q"]; L = state["L"]
        assert Q is not None
        return Q @ ((Q.mH @ b) / L)

    # use inverse if cached
    if "H_inv" in state:
        return state["H_inv"] @ b

    # use hessian
    H = state["H"]
    if use_lstsq: return _least_squares_solve(H, b)

    dir = None
    if dir is None: dir = _try_cholesky_solve(H, b)
    if dir is None: dir = _try_lu_solve(H, b)
    if dir is None: dir = _least_squares_solve(H, b)
    return dir

def _newton_get_H(state: dict[str, torch.Tensor | Any]):
    """used in most hessian-based modules. state is from ``_newton_update_state_``"""
    if "H_inv" in state:
        return DenseWithInverse(state["H"], state["H_inv"])

    if "L" in state:
        # Eigendecomposition has sligthly different solve_plus_diag
        # I am pretty sure it should be very close and it uses no solves
        # best way to test is to try cubic regularization with this
        return Eigendecomposition(state["L"], state["Q"], use_nystrom=False)

    return Dense(state["H"])

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
        eigval_fn (Callable | None, optional):
            function to apply to eigenvalues, for example ``torch.abs`` or ``lambda L: torch.clip(L, min=1e-8)``.
            If this is specified, eigendecomposition will be used to invert the hessian.
        update_freq (int, optional):
            updates hessian every ``update_freq`` steps.
        precompute_inverse (bool, optional):
            if ``True``, whenever hessian is computed, also computes the inverse. This is more efficient
            when ``update_freq`` is large. If ``None``, this is ``True`` if ``update_freq >= 10``.
        use_lstsq (bool, Optional):
            if True, least squares will be used to solve the linear system, this can prevent it from exploding
            when hessian is indefinite. If False, tries cholesky, if it fails tries LU, and then least squares.
            If ``eigval_fn`` is specified, eigendecomposition is always used and this argument is ignored.
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
            - ``"thoad"`` - uses ``thoad`` library, can be significantly faster than pytorch but limited operator coverage.

            Defaults to ``"batched_autograd"``.
        h (float, optional):
            finite difference step size if hessian is compute via finite-difference.
        inner (Chainable | None, optional): modules to apply hessian preconditioner to. Defaults to None.

    # See also

    * ``tz.m.NewtonCG``: uses a matrix-free conjugate gradient solver and hessian-vector products.
    useful for large scale problems as it doesn't form the full hessian.
    * ``tz.m.NewtonCGSteihaug``: trust region version of ``tz.m.NewtonCG``.
    * ``tz.m.ImprovedNewton``: Newton with additional rank one correction to the hessian, can be faster than Newton.
    * ``tz.m.InverseFreeNewton``: an inverse-free variant of Newton's method.
    * ``tz.m.quasi_newton``: large collection of quasi-newton methods that estimate the hessian.

    # Notes

    ## Implementation details

    ``(H + yI)⁻¹g`` is calculated by solving the linear system ``(H + yI)x = g``.
    The linear system is solved via cholesky decomposition, if that fails, LU decomposition, and if that fails, least squares. Least squares can be forced by setting ``use_lstsq=True``.

    Additionally, if ``eigval_fn`` is specified, eigendecomposition of the hessian is computed,
    ``eigval_fn`` is applied to the eigenvalues, and ``(H + yI)⁻¹`` is computed using the computed eigenvectors and transformed eigenvalues. This is more generally more computationally expensive but not by much.

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

    """
    def __init__(
        self,
        damping: float = 0,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        eigv_tol: float | None = None,
        truncate: int | None = None,
        update_freq: int = 1,
        precompute_inverse: bool | None = None,
        use_lstsq: bool = False,
        hessian_method: HessianMethod = "batched_autograd",
        h: float = 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['update_freq'], defaults["inner"]
        super().__init__(defaults, update_freq=update_freq, inner=inner)

    @torch.no_grad
    def update_states(self, objective, states, settings):
        fs = settings[0]

        precompute_inverse = fs["precompute_inverse"]
        if precompute_inverse is None:
            precompute_inverse = fs["__update_freq"] >= 10

        __, _, H = objective.hessian(hessian_method=fs["hessian_method"], h=fs["h"], at_x0=True)

        _newton_update_state_(
            state = self.global_state,
            H=H,
            damping = fs["damping"],
            eigval_fn = fs["eigval_fn"],
            eigv_tol = fs["eigv_tol"],
            truncate = fs["truncate"],
            precompute_inverse = precompute_inverse,
            use_lstsq = fs["use_lstsq"]
        )

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        updates = objective.get_updates()
        fs = settings[0]

        b = torch.cat([t.ravel() for t in updates])
        sol = _newton_solve(b=b, state=self.global_state, use_lstsq=fs["use_lstsq"])

        vec_to_tensors_(sol, updates)
        return objective

    def get_H(self,objective=...):
        return _newton_get_H(self.global_state)

