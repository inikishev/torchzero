from typing import Literal, overload

import torch
from scipy.sparse.linalg import LinearOperator, gcrotmk

from ...core import Chainable, Module, step
from ...utils import TensorList, vec_to_tensors
from ...utils.derivatives import hvp_fd_central, hvp_fd_forward


class ScipyNewtonCG(Module):
    """NewtonCG with scipy solvers (any from scipy.sparse.linalg)"""
    def __init__(
        self,
        solver = gcrotmk,
        hvp_method: Literal["fd_forward", "fd_central", "autograd"] = "autograd",
        h: float = 1e-3,
        warm_start=False,
        inner: Chainable | None = None,
        kwargs: dict | None = None,
    ):
        defaults = dict(hvp_method=hvp_method, solver=solver, h=h, warm_start=warm_start)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

        self._num_hvps = 0
        self._num_hvps_last_step = 0

        if kwargs is None: kwargs = {}
        self._kwargs = kwargs

    @torch.no_grad
    def apply(self, objective):
        params = TensorList(objective.params)
        closure = objective.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        fs = self.settings[params[0]]
        hvp_method = fs['hvp_method']
        solver = fs['solver']
        h = fs['h']
        warm_start = fs['warm_start']

        self._num_hvps_last_step = 0
        # ---------------------- Hessian vector product function --------------------- #
        device = params[0].device; dtype=params[0].dtype
        if hvp_method == 'autograd':
            grad = objective.get_grads(create_graph=True)

            def H_mm(x_np):
                self._num_hvps_last_step += 1
                x = vec_to_tensors(torch.as_tensor(x_np, device=device, dtype=dtype), grad)
                with torch.enable_grad():
                    Hvp = TensorList(torch.autograd.grad(grad, params, x, retain_graph=True))
                return torch.cat([t.ravel() for t in Hvp]).numpy(force=True)

        else:

            with torch.enable_grad():
                grad = objective.get_grads()

            if hvp_method == 'forward':
                def H_mm(x_np):
                    self._num_hvps_last_step += 1
                    x = vec_to_tensors(torch.as_tensor(x_np, device=device, dtype=dtype), grad)
                    Hvp = TensorList(hvp_fd_forward(closure, params, x, h=h, g_0=grad)[1])
                    return torch.cat([t.ravel() for t in Hvp]).numpy(force=True)

            elif hvp_method == 'central':
                def H_mm(x_np):
                    self._num_hvps_last_step += 1
                    x = vec_to_tensors(torch.as_tensor(x_np, device=device, dtype=dtype), grad)
                    Hvp = TensorList(hvp_fd_central(closure, params, x, h=h)[1])
                    return torch.cat([t.ravel() for t in Hvp]).numpy(force=True)

            else:
                raise ValueError(hvp_method)

        ndim = sum(p.numel() for p in params)
        H = LinearOperator(shape=(ndim,ndim), matvec=H_mm, rmatvec=H_mm) # type:ignore

        # -------------------------------- inner step -------------------------------- #
        objective = self.inner_step("inner", objective, must_exist=False)
        b = TensorList(objective.get_updates())

        # ---------------------------------- run cg ---------------------------------- #
        x0 = None
        if warm_start: x0 = self.global_state.get('x_prev', None) # initialized to 0 which is default anyway

        x_np = solver(H, b.to_vec().nan_to_num().numpy(force=True), x0=x0, **self._kwargs)
        if isinstance(x_np, tuple): x_np = x_np[0]

        if warm_start:
            self.global_state['x_prev'] = x_np

        objective.updates = vec_to_tensors(torch.as_tensor(x_np, device=device, dtype=dtype), params)

        self._num_hvps += self._num_hvps_last_step
        return objective

