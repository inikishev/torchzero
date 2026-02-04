from collections.abc import Callable
from typing import Any

import torch

from ...core import Chainable, Optimizer, Module, step, HVPMethod
from ...utils import TensorList
from ..quasi_newton import LBFGS


class NewtonSolver(Module):
    """Matrix free newton via with any custom solver (this is for testing, use NewtonCG or NystromPCG)."""
    def __init__(
        self,
        solver: Callable[[list[torch.Tensor]], Any] = lambda p: Optimizer(p, LBFGS()),
        maxiter=None,
        maxiter1=None,
        tol:float | None=1e-3,
        reg: float = 0,
        warm_start=True,
        hvp_method: HVPMethod = "autograd",
        reset_solver: bool = False,
        h: float= 1e-3,

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults['inner']
        super().__init__(defaults)

        self.set_child("inner", inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

    @torch.no_grad
    def apply(self, objective):

        params = TensorList(objective.params)
        closure = objective.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        solver_cls = settings['solver']
        maxiter = settings['maxiter']
        maxiter1 = settings['maxiter1']
        tol = settings['tol']
        hvp_method = settings['hvp_method']
        warm_start = settings['warm_start']
        h = settings['h']
        reset_solver = settings['reset_solver']

        self._num_hvps_last_step = 0

        # ---------------------- Hessian vector product function --------------------- #
        _, H_mv = objective.list_Hvp_function(hvp_method=hvp_method, h=h, at_x0=True)

        # -------------------------------- inner step -------------------------------- #
        objective = self.inner_step("inner", objective, must_exist=False)
        b = TensorList(objective.get_updates())

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
            Hx = H_mv(x).detach()
            # loss = (Hx-b).pow(2).global_mean()
            # if backward:
            #     solver.zero_grad()
            #     loss.backward(inputs=x)

            residual = Hx - b
            loss = residual.pow(2).global_mean()
            if backward:
                with torch.no_grad():
                    H_residual = H_mv(residual)
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

        objective.updates = x.detach()
        self._num_hvps += self._num_hvps_last_step
        return objective


