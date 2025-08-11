"""convenience submodule which allows to calculate a metric based on its string name,
used in many places"""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import torch

if TYPE_CHECKING:
    from .tensorlist import TensorList



class Metric(ABC):
    @abstractmethod
    def evaluate_global(self, x: "TensorList") -> torch.Tensor:
        """returns a global metric for a tensorlist"""

    @abstractmethod
    def evaluate_tensor(self, x: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
        """returns metric for a tensor"""

    def evaluate_list(self, x: "TensorList") -> "TensorList":
        """returns list of metrics for a tensorlist (possibly vectorized)"""
        return x.map(self.evaluate_tensor)


class _MAD(Metric):
    def evaluate_global(self, x): return x.abs().global_mean()
    def evaluate_tensor(self, x, dim=None, keepdim=False): return x.abs().mean(dim=dim, keepdim=keepdim)
    def evaluate_list(self, x): return x.abs().mean()

class _Std(Metric):
    def evaluate_global(self, x): return x.global_std()
    def evaluate_tensor(self, x, dim=None, keepdim=False): return x.std(dim=dim, keepdim=keepdim)
    def evaluate_list(self, x): return x.std()

class _Var(Metric):
    def evaluate_global(self, x): return x.global_var()
    def evaluate_tensor(self, x, dim=None, keepdim=False): return x.var(dim=dim, keepdim=keepdim)
    def evaluate_list(self, x): return x.var()

class _Sum(Metric):
    def evaluate_global(self, x): return x.global_sum()
    def evaluate_tensor(self, x, dim=None, keepdim=False): return x.sum(dim=dim, keepdim=keepdim)
    def evaluate_list(self, x): return x.sum()

class _Norm(Metric):
    def __init__(self, ord): self.ord = ord
    def evaluate_global(self, x): return x.global_vector_norm(self.ord)
    def evaluate_tensor(self, x, dim=None, keepdim=False):
        return torch.linalg.vector_norm(x, ord=self.ord, dim=dim, keepdim=keepdim) # pylint:disable=not-callable
    def evaluate_list(self, x): return x.norm(self.ord)

_METRIC_KEYS = Literal['mad', 'std', 'var', 'sum', 'l0', 'l1', 'l2', 'l3', 'l4', 'linf']
_METRICS: dict[_METRIC_KEYS, Metric] = {
    "mad": _MAD(),
    "std": _Std(),
    "var": _Var(),
    "sum": _Sum(),
    "l0": _Norm(0),
    "l1": _Norm(1),
    "l2": _Norm(2),
    "l3": _Norm(3),
    "l4": _Norm(4),
    "linf": _Norm(torch.inf),
}

Metrics = _METRIC_KEYS | float | torch.Tensor
def evaluate_metric(x: "torch.Tensor | TensorList", metric: Metrics) -> torch.Tensor:
    if isinstance(metric, (int, float, torch.Tensor)):
        if isinstance(x, torch.Tensor): return torch.linalg.vector_norm(x, ord=metric) # pylint:disable=not-callable
        return x.global_vector_norm(ord=float(metric))

    if isinstance(x, torch.Tensor): return _METRICS[metric].evaluate_tensor(x)
    return _METRICS[metric].evaluate_global(x)


def calculate_metric_list(x: "TensorList", metric: Metrics) -> "TensorList":
    if isinstance(metric, (int, float, torch.Tensor)):
        return x.norm(ord=float(metric))

    return _METRICS[metric].evaluate_list(x)
