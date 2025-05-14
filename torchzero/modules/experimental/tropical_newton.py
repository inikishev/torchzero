import warnings
from functools import partial
from typing import Literal
from collections.abc import Callable
import torch
import torchalgebras as ta

from ...core import Chainable, apply, Module
from ...utils import vec_to_tensors, TensorList
from ...utils.derivatives import (
    hessian_list_to_mat,
    hessian_mat,
    jacobian_and_hessian_wrt,
)

class MaxItersReached(Exception): pass
def tropical_lstsq(
    H: torch.Tensor,
    g: torch.Tensor,
    solver,
    maxiter,
    tol,
    algebra,
    verbose,
):
    algebra = ta.get_algebra(algebra)

    x = torch.zeros_like(g, requires_grad=True)
    best_x = x.detach().clone()
    best_loss = float('inf')
    opt = solver([x])

    niter = 0
    def closure(backward=True):
        nonlocal niter, best_x, best_loss
        if niter == maxiter: raise MaxItersReached
        niter += 1

        g_hat = algebra.mm(H, x)
        loss = torch.nn.functional.mse_loss(g_hat, g)
        if loss < best_loss:
            best_x = x.detach().clone()
            best_loss = loss.detach()

        if backward:
            opt.zero_grad()
            loss.backward()
        return loss

    loss = None
    prev_loss = float('inf')
    for i in range(maxiter):
        try:
            loss = opt.step(closure)
            if loss == 0: break
            if tol is not None and prev_loss - loss < tol: break
            prev_loss = loss
        except MaxItersReached:
            break

    if verbose: print(f'{best_loss = } after {niter} iters')
    return best_x.detach()

def tikhonov(H: torch.Tensor, reg: float, algebra: ta.Algebra = ta.TropicalSemiring()):
    if reg!=0:
        I = ta.AlgebraicTensor(torch.eye(H.size(-1), dtype=H.dtype, device=H.device), algebra)
        I = I * reg
        H = algebra.add(H, I.data)
    return H


class TropicalNewton(Module):
    """newton in other algebras, not practical because solving linear system is very hard."""
    def __init__(
        self,
        reg: float | None = None,
        hessian_method: Literal["autograd", "func", "autograd.functional"] = "autograd",
        vectorize: bool = True,
        solver=lambda p: torch.optim.LBFGS(p, line_search_fn='strong_wolfe'),
        maxiter=1000,
        tol: float | None = 1e-10,
        algebra: ta.Algebra | str = ta.TropicalSemiring('max'),
        verbose: bool = False,
        inner: Chainable | None = None,
    ):
        defaults = dict(reg=reg, hessian_method=hessian_method, vectorize=vectorize)
        super().__init__(defaults)

        self.lstsq_args:dict = dict(solver=solver, maxiter=maxiter, tol=tol, algebra=algebra, verbose=verbose)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, vars):
        params = TensorList(vars.params)
        closure = vars.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        reg = settings['reg']
        hessian_method = settings['hessian_method']
        vectorize = settings['vectorize']

        # ------------------------ calculate grad and hessian ------------------------ #
        if hessian_method == 'autograd':
            with torch.enable_grad():
                loss = vars.loss = vars.loss_approx = closure(False)
                g_list, H_list = jacobian_and_hessian_wrt([loss], params, batched=vectorize)
                g_list = [t[0] for t in g_list] # remove leading dim from loss
                vars.grad = g_list
                H = hessian_list_to_mat(H_list)

        elif hessian_method in ('func', 'autograd.functional'):
            strat = 'forward-mode' if vectorize else 'reverse-mode'
            with torch.enable_grad():
                g_list = vars.get_grad(retain_graph=True)
                H: torch.Tensor = hessian_mat(partial(closure, backward=False), params,
                                method=hessian_method, vectorize=vectorize, outer_jacobian_strategy=strat) # pyright:ignore[reportAssignmentType]

        else:
            raise ValueError(hessian_method)

        # -------------------------------- inner step -------------------------------- #
        if 'inner' in self.children:
            g_list = apply(self.children['inner'], list(g_list), params=params, grads=list(g_list), vars=vars)
        g = torch.cat([t.view(-1) for t in g_list])

        # ------------------------------- regulazition ------------------------------- #
        if reg is not None: H = tikhonov(H, reg)

        # ----------------------------------- solve ---------------------------------- #
        update = tropical_lstsq(H, g, **self.lstsq_args)
        vars.update = vec_to_tensors(update, params)
        return vars