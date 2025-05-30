from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

import torch

from ...core import Chainable, Module, apply, Modular
from ...utils import TensorList, as_tensorlist
from ...utils.derivatives import hvp
from ..quasi_newton import LBFGS

class NewtonSolver(Module):
    """Matrix free newton via with any custom solver (usually it is better to just use NewtonCG or NystromPCG is even better)"""
    def __init__(
        self,
        solver: Callable[[list[torch.Tensor]], Any] = lambda p: Modular(p, LBFGS()),
        maxiter=None,
        tol=1e-3,
        reg: float = 0,
        warm_start=True,
        inner: Chainable | None = None,
    ):
        defaults = dict(tol=tol, maxiter=maxiter, reg=reg, warm_start=warm_start, solver=solver)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, vars):
        params = TensorList(vars.params)
        closure = vars.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        solver_cls = settings['solver']
        maxiter = settings['maxiter']
        tol = settings['tol']
        reg = settings['reg']
        warm_start = settings['warm_start']

        # ---------------------- Hessian vector product function --------------------- #
        grad = vars.get_grad(create_graph=True)

        def H_mm(x):
            with torch.enable_grad():
                Hvp = TensorList(hvp(params, grad, x, create_graph=True))
                if reg != 0: Hvp = Hvp + (x*reg)
                return Hvp

        # -------------------------------- inner step -------------------------------- #
        b = as_tensorlist(grad)
        if 'inner' in self.children:
            b = as_tensorlist(apply(self.children['inner'], [g.clone() for g in grad], params=params, grads=grad, vars=vars))

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state('prev_x', params=params, cls=TensorList) # initialized to 0 which is default anyway
        if x0 is None: x = b.zeros_like().requires_grad_(True)
        else: x = x0.clone().requires_grad_(True)

        solver = solver_cls(x)
        def lstsq_closure(backward=True):
            Hx = H_mm(x)
            loss = (Hx-b).pow(2).global_mean()
            if backward:
                solver.zero_grad()
                loss.backward(inputs=x)
            return loss

        if maxiter is None: maxiter = b.global_numel()
        loss = None
        initial_loss = lstsq_closure(False)
        if initial_loss > tol:
            for i in range(maxiter):
                loss = solver.step(lstsq_closure)
                assert loss is not None
                if min(loss, loss/initial_loss) < tol: break

        # print(f'{loss = }')

        if warm_start:
            assert x0 is not None
            x0.copy_(x)

        vars.update = x.detach()
        return vars


