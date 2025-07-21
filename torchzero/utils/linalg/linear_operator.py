"""simplified version of https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html. This is used for trust regions."""
from functools import partial
import math
from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import cast, final

import torch

from ..torch_tools import tofloat, tonumpy, totensor

if find_spec('scipy') is not None:
    from scipy.sparse.linalg import LinearOperator as _ScipyLinearOperator
else:
    _ScipyLinearOperator = None

class LinearOperator(ABC):
    """this is used for trust region"""
    device: torch.types.Device
    dtype: torch.dtype

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement matvec")

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement rmatvec")

    def matmat(self, x: torch.Tensor) -> "LinearOperator":
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
        return len(self.size())

    @property
    def ndim(self) -> int:
        return self.ndimension()

    def _numpy_matvec(self, x, dtype=None):
        """returns Ax ndarray for scipy's LinearOperator"""
        Ax = self.matvec(totensor(x, device=self.device, dtype=self.dtype))
        Ax = tonumpy(Ax)
        if dtype is not None: Ax = Ax.astype(dtype)
        return Ax

    def _numpy_rmatvec(self, x, dtype=None):
        """returns Ax ndarray for scipy's LinearOperator"""
        Ax = self.rmatvec(totensor(x, device=self.device, dtype=self.dtype))
        Ax = tonumpy(Ax)
        if dtype is not None: Ax = Ax.astype(dtype)
        return Ax

    def scipy_linop(self, dtype=None):
        if _ScipyLinearOperator is None: raise ModuleNotFoundError("Scipy needs to be installed")
        return _ScipyLinearOperator(
            dtype=dtype,
            shape=self.size(),
            matvec=partial(self._numpy_matvec, dtype=dtype), # pyright:ignore[reportCallIssue]
            rmatvec=partial(self._numpy_rmatvec, dtype=dtype), # pyright:ignore[reportCallIssue]
        )

    def is_dense(self) -> bool:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement is_dense")

class Dense(LinearOperator):
    def __init__(self, x: torch.Tensor | LinearOperator):
        if isinstance(x, LinearOperator): x = x.to_tensor()
        self.A: torch.Tensor = x
        self.device = self.A.device
        self.dtype = self.A.dtype

    def matvec(self, x): return self.A.mv(x)
    def rmatvec(self, x): return self.A.mH.mv(x)

    def matmat(self, x): return Dense(self.A.mm(x))
    def rmatmat(self, x): return Dense(self.A.mH.mm(x))

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
    def is_dense(self): return True

class Diagonal(LinearOperator):
    def __init__(self, x: torch.Tensor):
        assert x.ndim == 1
        self.A: torch.Tensor = x
        self.device = self.A.device
        self.dtype = self.A.dtype

    def matvec(self, x): return self.A * x
    def rmatvec(self, x): return self.A * x

    def matmat(self, x): return Dense(x * self.A.unsqueeze(-1))
    def rmatmat(self, x): return Dense(x * self.A.unsqueeze(-1))

    def solve(self, x): return x/self.A
    def add(self, x): return Dense(x + self.A.diag_embed())
    def add_diagonal(self, x): return Diagonal(self.A + x)
    def diagonal(self): return self.A
    def inv(self): return Diagonal(1/self.A)
    def to_tensor(self): return self.A.diag_embed()
    def size(self): return (self.A.numel(), self.A.numel())
    def is_dense(self): return False

class ScaledIdentity(LinearOperator):
    def __init__(self, s: float | torch.Tensor = 1., shape=None, device=None, dtype=None):
        if isinstance(s, torch.Tensor):
            self.device = s.device
            self.dtype = s.dtype
        if device is not None: self.device = device
        if dtype is not None: self.dtype = dtype
        self.s = tofloat(s)
        self._shape = shape

    def matvec(self, x): return x * self.s
    def rmatvec(self, x): return x * self.s

    def matmat(self, x): return Dense(x * self.s)
    def rmatmat(self, x): return Dense(x * self.s)

    def solve(self, x): return x / self.s
    def add(self, x): return Dense(x + self.s)
    def add_diagonal(self, x):
        if isinstance(x, torch.Tensor) and x.numel() <= 1: x = x.item()
        if isinstance(x, (int,float)): return ScaledIdentity(x + self.s, shape=self._shape, device=self.device, dtype=self.dtype)
        return Diagonal(x + self.s)

    def diagonal(self):
        if self._shape is None: raise RuntimeError("Shape is None")
        return torch.full(self._shape, fill_value=self.s, device=self.device, dtype=self.dtype)

    def inv(self): return ScaledIdentity(1 / self.s, shape=self._shape, device=self.device, dtype=self.dtype)
    def to_tensor(self):
        if self._shape is None: raise RuntimeError("Shape is None")
        return torch.eye(*self.shape, device=self.device, dtype=self.dtype).mul_(self.s)

    def size(self):
        if self._shape is None: raise RuntimeError("Shape is None")
        return self._shape

    def __repr__(self):
        return f"ScaledIdentity(s={self.s}, shape={self._shape}, dtype={self.dtype}, device={self.device})"

    def is_dense(self): return False

class AtA(LinearOperator):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def matvec(self, x): return self.A.T.mv(self.A.mv(x))
    def rmatvec(self, x): return self.A.T.mv(self.A.mv(x))

    def matmat(self, x): return torch.linalg.multi_dot([self.A.T, self.A, x]) # pylint:disable=not-callable
    def rmatmat(self, x): return torch.linalg.multi_dot([self.A.T, self.A, x]) # pylint:disable=not-callable

    def is_dense(self): return False
    def to_tensor(self): return self.A.T @ self.A