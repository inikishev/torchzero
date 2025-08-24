from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

import torch

from ...core import Chainable, Modular, Module, apply_transform
from ...utils import TensorList, as_tensorlist
from ...utils.derivatives import hvp, hvp_fd_forward, hvp_fd_central
from ..quasi_newton import LBFGS


class NewtonSolver(Module):
    """Matrix free newton via with any custom solver (this is for testing, use NewtonCG or NystromPCG)."""
    def __init__(
        self,
        solver: Callable[[list[torch.Tensor]], Any] = lambda p: Modular(p, LBFGS()),
        maxiter=None,
        maxiter1=None,
        tol:float | None=1e-3,
        reg: float = 0,
        warm_start=True,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        reset_solver: bool = False,
        h: float= 1e-3,
        inner: Chainable | None = None,
    ):
        defaults = dict(tol=tol, h=h,reset_solver=reset_solver, maxiter=maxiter, maxiter1=maxiter1, reg=reg, warm_start=warm_start, solver=solver, hvp_method=hvp_method)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)
        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        solver_cls = settings['solver']
        maxiter = settings['maxiter']
        maxiter1 = settings['maxiter1']
        tol = settings['tol']
        reg = settings['reg']
        hvp_method = settings['hvp_method']
        warm_start = settings['warm_start']
        h = settings['h']
        reset_solver = settings['reset_solver']

        self._num_hvps_last_step = 0
        # ---------------------- Hessian vector product function --------------------- #
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                self._num_hvps_last_step += 1
                with torch.enable_grad():
                    Hvp = TensorList(hvp(params, grad, x, retain_graph=True))
                if reg != 0: Hvp = Hvp + (x*reg)
                return Hvp

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    self._num_hvps_last_step += 1
                    Hvp = TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad, normalize=True)[1])
                    if reg != 0: Hvp = Hvp + (x*reg)
                    return Hvp

            elif hvp_method == 'central':
                def H_mm(x):
                    self._num_hvps_last_step += 1
                    Hvp =  TensorList(hvp_fd_central(closure, params, x, h=h, normalize=True)[1])
                    if reg != 0: Hvp = Hvp + (x*reg)
                    return Hvp

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        b = as_tensorlist(grad)
        if 'inner' in self.children:
            b = as_tensorlist(apply_transform(self.children['inner'], [g.clone() for g in grad], params=params, grads=grad, var=var))

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.get_state(params, 'prev_x', cls=TensorList) # initialized to 0 which is default anyway
        if x0 is None: x = b.zeros_like().requires_grad_(True)
        else: x = x0.clone().requires_grad_(True)


        if 'solver' not in self.global_state:
            if maxiter1 is not None: maxiter = maxiter1
            solver = self.global_state['solver'] = solver_cls(x)
            self.global_state['x'] = x

        else:
            if reset_solver:
                solver = self.global_state['solver'] = solver_cls(x)
            else:
                solver_params = self.global_state['x']
                solver_params.set_(x)
                x = solver_params
                solver = self.global_state['solver']

        def lstsq_closure(backward=True):
            Hx = H_mm(x).detach()
            # loss = (Hx-b).pow(2).global_mean()
            # if backward:
            #     solver.zero_grad()
            #     loss.backward(inputs=x)

            residual = Hx - b
            loss = residual.pow(2).global_mean()
            if backward:
                with torch.no_grad():
                    H_residual = H_mm(residual)
                    n = residual.global_numel()
                    x.set_grad_((2.0 / n) * H_residual)

            return loss

        if maxiter is None: maxiter = b.global_numel()
        loss = None
        initial_loss = lstsq_closure(False) if tol is not None else None # skip unnecessary closure if tol is None
        if initial_loss is None or initial_loss > torch.finfo(b[0].dtype).eps:
            for i in range(maxiter):
                loss = solver.step(lstsq_closure)
                assert loss is not None
                if initial_loss is not None and loss/initial_loss < tol: break

        # print(f'{loss = }')

        if warm_start:
            assert x0 is not None
            x0.copy_(x)

        var.update = x.detach()
        self._num_hvps += self._num_hvps_last_step
        return var


