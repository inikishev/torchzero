from collections.abc import Callable
from typing import Literal, overload
import warnings
import torch

from ...utils import TensorList, as_tensorlist, generic_zeros_like, generic_vector_norm, generic_numel, vec_to_tensors
from ...utils.derivatives import hvp, hvp_fd_central, hvp_fd_forward

from ...core import Chainable, apply_transform, Module
from ...utils.linalg.solve import nystrom_sketch_and_solve, nystrom_pcg

class NystromSketchAndSolve(Module):
    def __init__(
        self,
        rank: int,
        reg: float = 1e-3,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        h=1e-3,
        inner: Chainable | None = None,
        seed: int | None = None,
    ):
        defaults = dict(rank=rank, reg=reg, hvp_method=hvp_method, h=h, seed=seed)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)

        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

        settings = self.settings[params[0]]
        rank = settings['rank']
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
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                with torch.enable_grad():
                    Hvp = hvp(params, grad, params.from_vec(x), retain_graph=True)
                    return torch.cat([t.ravel() for t in Hvp])

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    Hvp = hvp_fd_forward(closure, params, params.from_vec(x), h=h, g_0=grad, normalize=True)[1]
                    return torch.cat([t.ravel() for t in Hvp])

            elif hvp_method == 'central':
                def H_mm(x):
                    Hvp = hvp_fd_central(closure, params, params.from_vec(x), h=h, normalize=True)[1]
                    return torch.cat([t.ravel() for t in Hvp])

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
        if 'inner' in self.children:
            b = apply_transform(self.children['inner'], b, params=params, grads=grad, var=var)

        # ------------------------------ sketch&n&solve ------------------------------ #
        x = nystrom_sketch_and_solve(A_mm=H_mm, b=torch.cat([t.ravel() for t in b]), rank=rank, reg=reg, generator=generator)
        var.update = vec_to_tensors(x, reference=params)
        return var



class NystromPCG(Module):
    def __init__(
        self,
        sketch_size: int,
        maxiter=None,
        tol=1e-3,
        reg: float = 1e-6,
        hvp_method: Literal["forward", "central", "autograd"] = "autograd",
        h=1e-3,
        inner: Chainable | None = None,
        seed: int | None = None,
    ):
        defaults = dict(sketch_size=sketch_size, reg=reg, maxiter=maxiter, tol=tol, hvp_method=hvp_method, h=h, seed=seed)
        super().__init__(defaults,)

        if inner is not None:
            self.set_child('inner', inner)

    @torch.no_grad
    def step(self, var):
        params = TensorList(var.params)

        closure = var.closure
        if closure is None: raise RuntimeError('NewtonCG requires closure')

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
        if hvp_method == 'autograd':
            grad = var.get_grad(create_graph=True)

            def H_mm(x):
                with torch.enable_grad():
                    Hvp = hvp(params, grad, params.from_vec(x), retain_graph=True)
                    return torch.cat([t.ravel() for t in Hvp])

        else:

            with torch.enable_grad():
                grad = var.get_grad()

            if hvp_method == 'forward':
                def H_mm(x):
                    Hvp = hvp_fd_forward(closure, params, params.from_vec(x), h=h, g_0=grad, normalize=True)[1]
                    return torch.cat([t.ravel() for t in Hvp])

            elif hvp_method == 'central':
                def H_mm(x):
                    Hvp = hvp_fd_central(closure, params, params.from_vec(x), h=h, normalize=True)[1]
                    return torch.cat([t.ravel() for t in Hvp])

            else:
                raise ValueError(hvp_method)


        # -------------------------------- inner step -------------------------------- #
        b = var.get_update()
        if 'inner' in self.children:
            b = apply_transform(self.children['inner'], b, params=params, grads=grad, var=var)

        # ------------------------------ sketch&n&solve ------------------------------ #
        x = nystrom_pcg(A_mm=H_mm, b=torch.cat([t.ravel() for t in b]), sketch_size=sketch_size, reg=reg, tol=tol, maxiter=maxiter, x0_=None, generator=generator)
        var.update = vec_to_tensors(x, reference=params)
        return var


