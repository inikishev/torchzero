from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from ...utils import TensorList, tonumpy
from ...utils.derivatives import (
    flatten_jacobian,
    jacobian_and_hessian_mat_wrt,
    jacobian_wrt,
)


class WrapperBase(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    @torch.no_grad
    def _f(self, x: np.ndarray, params: list[torch.Tensor], closure) -> float:
        # set params to x
        params = TensorList(params)
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        return float(closure(False))

    @torch.no_grad
    def _fs(self, x: np.ndarray, params: list[torch.Tensor], closure) -> np.ndarray:
        # set params to x
        params = TensorList(params)
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        return tonumpy(closure(False)).reshape(-1)


    @torch.no_grad
    def _f_g(self, x: np.ndarray, params: list[torch.Tensor], closure) -> tuple[float, np.ndarray]:
        # set params to x
        params = TensorList(params)
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # compute value and derivatives
        with torch.enable_grad():
            value = closure()
            g = params.grad.fill_none(reference=params).to_vec()
        return float(value), g.numpy(force=True)

    @torch.no_grad
    def _f_g_H(self, x: np.ndarray, params: list[torch.Tensor], closure) -> tuple[float, np.ndarray, np.ndarray]:
        # set params to x
        params = TensorList(params)
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # compute value and derivatives
        with torch.enable_grad():
            value = closure(False)
            g, H = jacobian_and_hessian_mat_wrt([value], wrt = params)
        return float(value), g.numpy(force=True), H.numpy(force=True)

    @torch.no_grad
    def _f_g_Hvp(self, x: np.ndarray, v: np.ndarray, params: list[torch.Tensor], closure) -> tuple[float, np.ndarray, np.ndarray]:
        # set params to x
        params = TensorList(params)
        params.from_vec_(torch.from_numpy(x).to(device = params[0].device, dtype=params[0].dtype, copy=False))

        # compute value and derivatives
        with torch.enable_grad():
            value = closure(False)
            grad = torch.autograd.grad(value, params, create_graph=True, allow_unused=True, materialize_grads=True)
            flat_grad = torch.cat([i.reshape(-1) for i in grad])
            Hvp = torch.autograd.grad(flat_grad, params, torch.as_tensor(v, device=flat_grad.device, dtype=flat_grad.dtype))[0]

        return float(value), flat_grad.numpy(force=True), Hvp.numpy(force=True)

    def _get_params(self) -> list[torch.Tensor]:
        return [p for g in self.param_groups for p in g["params"]]

    def _get_per_parameter_lb_ub(self):
        # get per-parameter lb and ub
        lb = []
        ub = []
        for group in self.param_groups:
            lb.extend([group["lb"]] * len(group["params"]))
            ub.extend([group["ub"]] * len(group["params"]))

        return lb, ub

    def _get_bounds(self):

        # get per-parameter lb and ub
        lb, ub = self._get_per_parameter_lb_ub()
        if all(i is None for i in lb) and all(i is None for i in ub): return None

        params = self._get_params()
        bounds = []
        for p, l, u in zip(params, lb, ub):
            bounds.extend([(l, u)] * p.numel())

        return bounds

    def _get_lb_ub(self, ld:dict | None = None, ud: dict | None = None):
        if ld is None: ld = {}
        if ud is None: ud = {}

        # get per-parameter lb and ub
        lb, ub = self._get_per_parameter_lb_ub()

        params = self._get_params()
        lb_list = []
        ub_list = []
        for p, l, u in zip(params, lb, ub):
            if l in ld: l = ld[l]
            if u in ud: l = ud[u]
            lb_list.extend([l] * p.numel())
            ub_list.extend([u] * p.numel())

        return lb_list, ub_list

    @abstractmethod
    def step(self, closure) -> Any: # pyright:ignore[reportIncompatibleMethodOverride] # pylint:disable=signature-differs
        ...