import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch

from ...core import Chainable, Module, apply_transform, Var
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import (
    flatten_jacobian,
    hessian_mat,
    hvp,
    hvp_fd_central,
    hvp_fd_forward,
    jacobian_and_hessian_wrt,
)
from ...utils.linalg.linear_operator import DenseWithInverse, Dense

def _lu_solve(H: torch.Tensor, g: torch.Tensor):
    try:
        x, info = torch.linalg.solve_ex(H, g) # pylint:disable=not-callable
        if info == 0: return x
        return None
    except RuntimeError:
        return None

def _cholesky_solve(H: torch.Tensor, g: torch.Tensor):
    x, info = torch.linalg.cholesky_ex(H) # pylint:disable=not-callable
    if info == 0:
        g.unsqueeze_(1)
        return torch.cholesky_solve(g, x)
    return None

def _least_squares_solve(H: torch.Tensor, g: torch.Tensor):
    return torch.linalg.lstsq(H, g)[0] # pylint:disable=not-callable

def _eigh_solve(H: torch.Tensor, g: torch.Tensor, tfm: Callable | None, search_negative: bool):
    try:
        L, Q = torch.linalg.eigh(H) # pylint:disable=not-callable
        if tfm is not None: L = tfm(L)
        if search_negative and L[0] < 0:
            neg_mask = L < 0
            Q_neg = Q[:, neg_mask] * L[neg_mask]
            return (Q_neg * (g @ Q_neg).sign()).mean(1)

        return Q @ ((Q.mH @ g) / L)

    except torch.linalg.LinAlgError:
        return None


def _get_loss_grad_and_hessian(var: Var, hessian_method:str, vectorize:bool):
    """returns (loss, g_list, H). Also sets var.loss and var.grad.
    If hessian_method isn't 'autograd', loss is not set and returned as None"""
    closure = var.closure
    if closure is None:
        raise RuntimeError("Second order methods requires a closure to be provided to the `step` method.")

    params = var.params

    # ------------------------ calculate grad and hessian ------------------------ #
    loss = None
    if hessian_method == 'autograd':
        with torch.enable_grad():
            loss = var.loss = var.loss_approx = closure(False)
            g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=vectorize)
            g_list = [t[0] for t in g_list] # remove leading dim from loss
            var.grad = g_list
            H = flatten_jacobian(H_list)

    elif hessian_method in ('func', 'autograd.functional'):
        strat = 'forward-mode' if vectorize else 'reverse-mode'
        with torch.enable_grad():
            g_list = var.get_grad(retain_graph=True)
            H = hessian_mat(partial(closure, backward=False), params,
                            method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

    else:
        raise ValueError(hessian_method)

    return loss, g_list, H

def _newton_step(var: Var, H: torch.Tensor, damping:float, inner: Module | None, H_tfm, eigval_fn, use_lstsq:bool, g_proj: Callable | None = None) -> torch.Tensor:
    """returns the update tensor, then do vec_to_tensor(update, params)"""
    params = var.params

    if damping != 0:
        H = H + torch.eye(H.size(-1), dtype=H.dtype, device=H.device).mul_(damping)

    # -------------------------------- inner step -------------------------------- #
    update = var.get_update()
    if inner is not None:
        update = apply_transform(inner, update, params=params, grads=var.grad, loss=var.loss, var=var)

    g = torch.cat([t.ravel() for t in update])
    if g_proj is not None: g = g_proj(g)

    # ----------------------------------- solve ---------------------------------- #
    update = None

    if H_tfm is not None:
        ret = H_tfm(H, g)

        if isinstance(ret, torch.Tensor):
            update = ret

        else: # returns (H, is_inv)
            H, is_inv = ret
            if is_inv: update = H @ g

    if eigval_fn is not None:
        update = _eigh_solve(H, g, eigval_fn, search_negative=False)

    if update is None and use_lstsq: update = _least_squares_solve(H, g)
    if update is None: update = _cholesky_solve(H, g)
    if update is None: update = _lu_solve(H, g)
    if update is None: update = _least_squares_solve(H, g)

    return update

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

class Newton(Module):
    """Exact newton's method via autograd.

    Newton's method produces a direction jumping to the stationary point of quadratic approximation of the target function.
    The update rule is given by ``(H + yI)⁻¹g``, where ``H`` is the hessian and ``g`` is the gradient, ``y`` is the ``damping`` parameter.
    ``g`` can be output of another module, if it is specifed in ``inner`` argument.

    Note:
        In most cases Newton should be the first module in the chain because it relies on autograd. Use the :code:`inner` argument if you wish to apply Newton preconditioning to another module's output.

    Note:
        This module requires the a closure passed to the optimizer step,
        as it needs to re-evaluate the loss and gradients for calculating the hessian.
        The closure must accept a ``backward`` argument (refer to documentation).

    Args:
        damping (float, optional): tikhonov regularizer value. Set this to 0 when using trust region. Defaults to 0.
        search_negative (bool, Optional):
            if True, whenever a negative eigenvalue is detected,
            search direction is proposed along weighted sum of eigenvectors corresponding to negative eigenvalues.
        use_lstsq (bool, Optional):
            if True, least squares will be used to solve the linear system, this may generate reasonable directions
            when hessian is not invertible. If False, tries cholesky, if it fails tries LU, and then least squares.
            If ``eigval_fn`` is specified, eigendecomposition will always be used to solve the linear system and this
            argument will be ignored.
        hessian_method (str):
            how to calculate hessian. Defaults to "autograd".
        vectorize (bool, optional):
            whether to enable vectorized hessian. Defaults to True.
        H_tfm (Callable | None, optional):
            optional hessian transforms, takes in two arguments - `(hessian, gradient)`.

            must return either a tuple: `(hessian, is_inverted)` with transformed hessian and a boolean value
            which must be True if transform inverted the hessian and False otherwise.

            Or it returns a single tensor which is used as the update.

            Defaults to None.
        eigval_fn (Callable | None, optional):
            optional eigenvalues transform, for example ``torch.abs`` or ``lambda L: torch.clip(L, min=1e-8)``.
            If this is specified, eigendecomposition will be used to invert the hessian.
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

    The first modification to handle non-convexity is to modify the eignevalues to be positive,
    for example by setting ``eigval_fn = lambda L: L.abs().clip(min=1e-4)``.

    Second modification is ``search_negative=True``, which will search along a negative curvature direction if one is detected.
    This also requires an eigendecomposition.

    The Newton direction can also be forced to be a descent direction by using ``tz.m.GradSign()`` or ``tz.m.Cautious``,
    but that may be significantly less efficient.

    # Examples:

    Newton's method with backtracking line search

    ```py
    opt = tz.Modular(
        model.parameters(),
        tz.m.Newton(),
        tz.m.Backtracking()
    )
    ```

    Newton preconditioning applied to momentum

    ```py
    opt = tz.Modular(
        model.parameters(),
        tz.m.Newton(inner=tz.m.EMA(0.9)),
        tz.m.LR(0.1)
    )
    ```

    Diagonal newton example. This will still evaluate the entire hessian so it isn't efficient,
    but if you wanted to see how diagonal newton behaves or compares to full newton, you can use this.

    ```py
    opt = tz.Modular(
        model.parameters(),
        tz.m.Newton(H_tfm = lambda H, g: g/H.diag()),
        tz.m.Backtracking()
    )
    ```

    """
    def __init__(
        self,
        damping: float = 0,
        use_lstsq: bool = False,
        update_freq: int = 1,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        H_tfm: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, bool]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eigval_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        inner: Chainable | None = None,
    ):
        defaults = dict(damping=damping, hessian_method=hessian_method, use_lstsq=use_lstsq, vectorize=vectorize, H_tfm=H_tfm, eigval_fn=eigval_fn, update_freq=update_freq)
        super().__init__(defaults)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def update(self, var):
        step = self.global_state.get('step', 0)
        self.global_state['step'] = step + 1

        if step % self.defaults['update_freq'] == 0:
            loss, g_list, self.global_state['H'] = _get_loss_grad_and_hessian(
                var, self.defaults['hessian_method'], self.defaults['vectorize']
            )

    @torch.no_grad
    def apply(self, var):
        params = var.params
        update = _newton_step(
            var=var,
            H = self.global_state["H"],
            damping=self.defaults["damping"],
            inner=self.children.get("inner", None),
            H_tfm=self.defaults["H_tfm"],
            eigval_fn=self.defaults["eigval_fn"],
            use_lstsq=self.defaults["use_lstsq"],
        )

        var.update = vec_to_tensors(update, params)

        return var

    def get_H(self,var=...):
        return _get_H(self.global_state["H"], self.defaults["eigval_fn"])

