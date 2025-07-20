"""very simplified version of https://linear-operator.readthedocs.io/en/latest/linear_operator.html. This is used for trust regions."""
import math
from abc import ABC, abstractmethod
from typing import final, cast

import torch

class LinearOperator(ABC):
    """this is used for trust region"""

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement matvec")

    def matmul(self, x: torch.Tensor) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement matmul")

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement solve")

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement update")

    def add(self, x: torch.Tensor) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement add")

    def __add__(self, x: torch.Tensor) -> "LinearOperator":
        return self.add(x)

    def add_diagonal(self, x: torch.Tensor | float) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement add_diagonal")

    def diagonal(self) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement diagonal")

    def inv(self) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement inverse")

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement to_tensor")

    def to_dense(self) -> "Dense":
        return Dense(self)

    def size(self) -> tuple[int, ...]:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement size")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size()

    def numel(self) -> int:
        return math.prod(self.size())

    def ndimension(self) -> int:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement ndimension")

    @property
    def ndim(self) -> int:
        return self.ndimension()

class Dense(LinearOperator):
    def __init__(self, x: torch.Tensor | LinearOperator):
        if isinstance(x, LinearOperator): x = x.to_tensor()
        self.A: torch.Tensor = x

    def matvec(self, x): return self.A @ x
    def matmul(self, x): return Dense(self.A @ x)
    def solve(self, x): return torch.linalg.solve(self.A, x) # pylint:disable=not-callable
    def add(self, x): return Dense(self.A + x)
    def add_diagonal(self, x):
        if isinstance(x, torch.Tensor) and x.numel() <= 1: x = x.item()
        if isinstance(x, (int,float)): x = torch.full((self.shape[0],), fill_value=x, device=self.A.device, dtype=self.A.dtype)
        return Dense(self.A + torch.diag_embed(x))
    def diagonal(self): return self.A.diagonal()
    def inv(self): return Dense(torch.linalg.inv(self.A)) # pylint:disable=not-callable
    def to_tensor(self): return self.A
    def size(self): return self.A.size()
    def ndimension(self): return self.A.ndimension()


class Diagonal(LinearOperator):
    def __init__(self, x: torch.Tensor):
        assert x.ndim == 1
        self.A: torch.Tensor = x

    def matvec(self, x): return self.A * x
    def matmul(self, x): return Dense(x * self.A.unsqueeze(-1))
    def solve(self, x): return x/self.A
    def add(self, x): return Dense(self.A + x)
    def add_diagonal(self, x): return Diagonal(self.A + x)
    def diagonal(self): return self.A
    def inv(self): return Diagonal(1/self.A)
    def to_tensor(self): return self.A.diag_embed()
    def size(self): return (self.A.numel(), self.A.numel())
    def ndimension(self): return self.A.ndimension() + 1

class MtM(LinearOperator):
    def __init__(self, M: torch.Tensor):
        self.M = M

    def matvec(self, x):
        return self.M.T @ (self.M @ x)