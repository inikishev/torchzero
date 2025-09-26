from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from ...core import Chainable, DerivativesMethod, Objective, Transform
from ...utils import TensorList, vec_to_tensors


class HigherOrderMethodBase(Transform, ABC):
    def __init__(self, defaults: dict | None = None, derivatives_method: DerivativesMethod = 'batched_autograd'):
        self._derivatives_method: DerivativesMethod = derivatives_method
        super().__init__(defaults)

    @abstractmethod
    def one_iteration(
        self,
        x: torch.Tensor,
        evaluate: Callable[[torch.Tensor, int], tuple[torch.Tensor, ...]],
        objective: Objective,
        setting: Mapping[str, Any],
    ) -> torch.Tensor:
        """"""

    @torch.no_grad
    def apply_states(self, objective, states, settings):
        params = TensorList(objective.params)

        closure = objective.closure
        if closure is None: raise RuntimeError('MultipointNewton requires closure')
        derivatives_method = self._derivatives_method

        def evaluate(x, order) -> tuple[torch.Tensor, ...]:
            """order=0 - returns (loss,), order=1 - returns (loss, grad), order=2 - returns (loss, grad, hessian), etc."""
            return objective.derivatives_at(x, order, method=derivatives_method)

        x = torch.cat([p.ravel() for p in params])
        dir = self.one_iteration(x, evaluate, objective, settings[0])
        objective.updates = vec_to_tensors(dir, objective.params)
        return objective

def _inv(A: torch.Tensor, lstsq:bool) -> torch.Tensor:
    if lstsq: return torch.linalg.pinv(A) # pylint:disable=not-callable
    A_inv, info = torch.linalg.inv_ex(A) # pylint:disable=not-callable
    if info == 0: return A_inv
    return torch.linalg.pinv(A) # pylint:disable=not-callable

def _solve(A: torch.Tensor, b: torch.Tensor, lstsq: bool) -> torch.Tensor:
    if lstsq: return torch.linalg.lstsq(A, b).solution # pylint:disable=not-callable
    sol, info = torch.linalg.solve_ex(A, b) # pylint:disable=not-callable
    if info == 0: return sol
    return torch.linalg.lstsq(A, b).solution # pylint:disable=not-callable

# 3f 2J 3 solves
def sixth_order_3p(x:torch.Tensor, f, f_j, lstsq:bool=False):
    f_x, J_x = f_j(x)

    y = x - _solve(J_x, f_x, lstsq=lstsq)
    f_y, J_y = f_j(y)

    z = y - _solve(J_y, f_y, lstsq=lstsq)
    f_z = f(z)

    return y - _solve(J_y, f_y+f_z, lstsq=lstsq)

class SixthOrder3P(HigherOrderMethodBase):
    """Sixth-order iterative method.

    Abro, Hameer Akhtar, and Muhammad Mujtaba Shaikh. "A new time-efficient and convergent nonlinear solver." Applied Mathematics and Computation 355 (2019): 516-536.
    """
    def __init__(self, lstsq: bool=False, derivatives_method: DerivativesMethod = 'batched_autograd'):
        defaults=dict(lstsq=lstsq)
        super().__init__(defaults=defaults, derivatives_method=derivatives_method)

    @torch.no_grad
    def one_iteration(self, x, evaluate, objective, setting):
        def f(x): return evaluate(x, 1)[1]
        def f_j(x): return evaluate(x, 2)[1:]
        x_star = sixth_order_3p(x, f, f_j, setting['lstsq'])
        return x - x_star

# I don't think it works (I tested root finding with this and it goes all over the place)
# I double checked it multiple times
# def sixth_order_im1(x:torch.Tensor, f, f_j, lstsq:bool=False):
#     f_x, J_x = f_j(x)
#     J_x_inv = _inv(J_x, lstsq=lstsq)

#     y = x - J_x_inv @ f_x
#     f_y, J_y = f_j(y)

#     z = x - 2 * _solve(J_x + J_y, f_x, lstsq=lstsq)
#     f_z = f(z)

#     I = torch.eye(J_y.size(0), device=J_y.device, dtype=J_y.dtype)
#     term1 = (7/2)*I
#     term2 = 4 * (J_x_inv@J_y)
#     term3 = (3/2) * (J_x_inv @ (J_y@J_y))

#     return z - (term1 - term2 + term3) @ J_x_inv @ f_z

# class SixthOrderIM1(HigherOrderMethodBase):
#     """sixth-order iterative method https://www.mdpi.com/2504-3110/8/3/133

#     """
#     def __init__(self, lstsq: bool=False, vectorize: bool = True):
#         defaults=dict(lstsq=lstsq)
#         super().__init__(defaults=defaults, vectorize=vectorize)

#     def iteration(self, x, evaluate, var):
#         settings = self.defaults
#         lstsq = settings['lstsq']
#         def f(x): return evaluate(x, 1)[1]
#         def f_j(x): return evaluate(x, 2)[1:]
#         x_star = sixth_order_im1(x, f, f_j, lstsq)
#         return x - x_star

# 5f 5J 3 solves
def sixth_order_5p(x:torch.Tensor, f_j, lstsq:bool=False):
    f_x, J_x = f_j(x)
    y = x - _solve(J_x, f_x, lstsq=lstsq)

    f_y, J_y = f_j(y)
    f_xy2, J_xy2 = f_j((x + y) / 2)

    A = J_x + 2*J_xy2 + J_y

    z = y - 4*_solve(A, f_y, lstsq=lstsq)
    f_z, J_z = f_j(z)

    f_xz2, J_xz2 = f_j((x + z) / 2)
    B = J_x + 2*J_xz2 + J_z

    return z - 4*_solve(B, f_z, lstsq=lstsq)

class SixthOrder5P(HigherOrderMethodBase):
    """Argyros, Ioannis K., et al. "Extended convergence for two sixth order methods under the same weak conditions." Foundations 3.1 (2023): 127-139."""
    def __init__(self, lstsq: bool=False, derivatives_method: DerivativesMethod = 'batched_autograd'):
        defaults=dict(lstsq=lstsq)
        super().__init__(defaults=defaults, derivatives_method=derivatives_method)

    @torch.no_grad
    def one_iteration(self, x, evaluate, objective, setting):
        def f_j(x): return evaluate(x, 2)[1:]
        x_star = sixth_order_5p(x, f_j, setting['lstsq'])
        return x - x_star

# 2f 1J 2 solves
def two_point_newton(x: torch.Tensor, f, f_j, lstsq:bool=False):
    """third order convergence"""
    f_x, J_x = f_j(x)
    y = x - _solve(J_x, f_x, lstsq=lstsq)
    f_y = f(y)
    return x - _solve(J_x, f_x + f_y, lstsq=lstsq)

class TwoPointNewton(HigherOrderMethodBase):
    """two-point Newton method with frozen derivative with third order convergence.

    Sharma, Janak Raj, and Deepak Kumar. "A fast and efficient composite Newton–Chebyshev method for systems of nonlinear equations." Journal of Complexity 49 (2018): 56-73."""
    def __init__(self, lstsq: bool=False, derivatives_method: DerivativesMethod = 'batched_autograd'):
        defaults=dict(lstsq=lstsq)
        super().__init__(defaults=defaults, derivatives_method=derivatives_method)

    @torch.no_grad
    def one_iteration(self, x, evaluate, objective, setting):
        def f(x): return evaluate(x, 1)[1]
        def f_j(x): return evaluate(x, 2)[1:]
        x_star = two_point_newton(x, f, f_j, setting['lstsq'])
        return x - x_star

#3f 2J 1inv
def sixth_order_3pm2(x:torch.Tensor, f, f_j, lstsq:bool=False):
    f_x, J_x = f_j(x)
    J_x_inv = _inv(J_x, lstsq=lstsq)
    y = x - J_x_inv @ f_x
    f_y, J_y = f_j(y)

    I = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
    term = (2*I - J_x_inv @ J_y) @ J_x_inv
    z = y - term @ f_y

    return z - term @ f(z)


class SixthOrder3PM2(HigherOrderMethodBase):
    """Wang, Xiaofeng, and Yang Li. "An efficient sixth-order Newton-type method for solving nonlinear systems." Algorithms 10.2 (2017): 45."""
    def __init__(self, lstsq: bool=False, derivatives_method: DerivativesMethod = 'batched_autograd'):
        defaults=dict(lstsq=lstsq)
        super().__init__(defaults=defaults, derivatives_method=derivatives_method)

    @torch.no_grad
    def one_iteration(self, x, evaluate, objective, setting):
        def f_j(x): return evaluate(x, 2)[1:]
        def f(x): return evaluate(x, 1)[1]
        x_star = sixth_order_3pm2(x, f, f_j, setting['lstsq'])
        return x - x_star

