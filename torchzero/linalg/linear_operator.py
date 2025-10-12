"""This is mainly used for trust regions. In some cases certain operations are relaxed, e.g. eigenvalue shift instead of
adding diagonal when it isn't tractable, to make it work with Levenberg-Marquadt.
"""
import math
from abc import ABC, abstractmethod
from functools import partial
from importlib.util import find_spec
from typing import cast, final

import torch

from ..utils.torch_tools import tofloat, tonumpy, totensor
from .solve import nystrom_sketch_and_solve

if find_spec('scipy') is not None:
    from scipy.sparse.linalg import LinearOperator as _ScipyLinearOperator
else:
    _ScipyLinearOperator = None

class LinearOperator(ABC):
    device: torch.types.Device
    dtype: torch.dtype | None

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement matvec")

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement rmatvec")

    def matmat(self, X: torch.Tensor) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement matmat")

    def rmatmat(self, X: torch.Tensor) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement rmatmat")

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement solve")

    def solve_plus_diag(self, b: torch.Tensor, diag: int | float | torch.Tensor) -> torch.Tensor:
        return self.add_diagonal(diag).solve(b)

    def solve_bounded(self, b: torch.Tensor, bound:float, ord:float=2) -> torch.Tensor:
        """solve with a norm bound on x"""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement solve_bounded")

    # def update(self, *args, **kwargs) -> None:
    #     raise NotImplementedError(f"{self.__class__.__name__} doesn't implement update")

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

    def transpose(self) -> "LinearOperator":
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement transpose")

    @property
    def T(self): return self.transpose()

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement to_tensor")

    def to_dense(self) -> "Dense":
        return Dense(self) # calls to_tensor

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

def _solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # should I keep this or separate solve and lstsq?
    sol, info = torch.linalg.solve_ex(A, b) # pylint:disable=not-callable
    if info == 0: return sol
    return torch.linalg.lstsq(A, b).solution # pylint:disable=not-callable

def _inv(A: torch.Tensor) -> torch.Tensor:
    sol, info = torch.linalg.inv_ex(A) # pylint:disable=not-callable
    if info == 0: return sol
    return torch.linalg.pinv(A) # pylint:disable=not-callable


class Dense(LinearOperator):
    def __init__(self, A: torch.Tensor | LinearOperator):
        if isinstance(A, LinearOperator): A = A.to_tensor()
        self.A: torch.Tensor = A
        self.device = self.A.device
        self.dtype = self.A.dtype

    def matvec(self, x): return self.A.mv(x)
    def rmatvec(self, x): return self.A.mH.mv(x)

    def matmat(self, X): return Dense(self.A.mm(X))
    def rmatmat(self, X): return Dense(self.A.mH.mm(X))

    def solve(self, b): return _solve(self.A, b)

    def add(self, x): return Dense(self.A + x)
    def add_diagonal(self, x):
        if isinstance(x, torch.Tensor) and x.numel() <= 1: x = x.item()
        if isinstance(x, (int,float)): x = torch.full((self.shape[0],), fill_value=x, device=self.A.device, dtype=self.A.dtype)
        return Dense(self.A + torch.diag_embed(x))
    def diagonal(self): return self.A.diagonal()
    def inv(self): return Dense(_inv(self.A)) # pylint:disable=not-callable
    def to_tensor(self): return self.A
    def size(self): return self.A.size()
    def is_dense(self): return True
    def transpose(self): return Dense(self.A.mH)

class SPD(Dense):
    def solve(self, b: torch.Tensor):
        L, info = torch.linalg.cholesky_ex(self.A) # pylint:disable=not-callable
        return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)


class DenseInverse(LinearOperator):
    """Represents inverse of a dense matrix A."""
    def __init__(self, A_inv: torch.Tensor):
        self.A_inv: torch.Tensor = A_inv
        self.device = self.A_inv.device
        self.dtype = self.A_inv.dtype

    def matvec(self, x): return _solve(self.A_inv, x) # pylint:disable=not-callable
    def rmatvec(self, x): return _solve(self.A_inv.mH, x) # pylint:disable=not-callable

    def matmat(self, X): return Dense(_solve(self.A_inv, X)) # pylint:disable=not-callable
    def rmatmat(self, X): return Dense(_solve(self.A_inv.mH, X)) # pylint:disable=not-callable

    def solve(self, b): return self.A_inv.mv(b)

    def inv(self): return Dense(self.A_inv) # pylint:disable=not-callable
    def to_tensor(self): return _inv(self.A_inv) # pylint:disable=not-callable
    def size(self): return self.A_inv.size()
    def is_dense(self): return True
    def transpose(self): return DenseInverse(self.A_inv.mH)

class DenseWithInverse(Dense):
    """Represents a matrix where both the matrix and the inverse are known.

    ``matmat``, ``rmatmat``, ``add`` and ``add_diagonal`` will return a Dense matrix, inverse will be lost.
    """
    def __init__(self, A: torch.Tensor, A_inv: torch.Tensor):
        super().__init__(A)
        self.A_inv: torch.Tensor = A_inv

    def solve(self, b): return self.A_inv.mv(b)
    def inv(self): return DenseWithInverse(self.A_inv, self.A) # pylint:disable=not-callable
    def transpose(self): return DenseWithInverse(self.A.mH, self.A_inv.mH)

class Diagonal(LinearOperator):
    def __init__(self, x: torch.Tensor):
        assert x.ndim == 1
        self.A: torch.Tensor = x
        self.device = self.A.device
        self.dtype = self.A.dtype

    def matvec(self, x): return self.A * x
    def rmatvec(self, x): return self.A * x

    def matmat(self, X): return Dense(X * self.A.unsqueeze(-1))
    def rmatmat(self, X): return Dense(X * self.A.unsqueeze(-1))

    def solve(self, b): return b/self.A

    def add(self, x): return Dense(x + self.A.diag_embed())
    def add_diagonal(self, x): return Diagonal(self.A + x)
    def diagonal(self): return self.A
    def inv(self): return Diagonal(1/self.A)
    def to_tensor(self): return self.A.diag_embed()
    def size(self): return (self.A.numel(), self.A.numel())
    def is_dense(self): return False
    def transpose(self): return Diagonal(self.A)

class ScaledIdentity(LinearOperator):
    def __init__(self, s: float | torch.Tensor = 1., shape=None, device=None, dtype=None):
        self.device = self.dtype = None

        if isinstance(s, torch.Tensor):
            self.device = s.device
            self.dtype = s.dtype

        if device is not None: self.device = device
        if dtype is not None: self.dtype = dtype

        self.s = tofloat(s)
        self._shape = shape

    def matvec(self, x): return x * self.s
    def rmatvec(self, x): return x * self.s

    def matmat(self, X): return Dense(X * self.s)
    def rmatmat(self, X): return Dense(X * self.s)

    def solve(self, b): return b / self.s
    def solve_bounded(self, b, bound, ord = 2):
        b_norm = torch.linalg.vector_norm(b, ord=ord) # pylint:disable=not-callable
        sol = b / self.s
        sol_norm = b_norm / abs(self.s)

        if sol_norm > bound:
            if not math.isfinite(sol_norm):
                if b_norm > bound: return b * (bound / b_norm)
                return b
            return sol * (bound / sol_norm)

        return sol

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
    def transpose(self): return ScaledIdentity(self.s, shape=self.shape, device=self.device, dtype=self.dtype)


class AtA(LinearOperator):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def matvec(self, x): return self.A.mH.mv(self.A.mv(x))
    def rmatvec(self, x): return self.matvec(x)

    def matmat(self, X): return Dense(torch.linalg.multi_dot([self.A.mH, self.A, X])) # pylint:disable=not-callable
    def rmatmat(self, X): return Dense(torch.linalg.multi_dot([self.A.mH, self.A, X])) # pylint:disable=not-callable

    def is_dense(self): return False
    def to_tensor(self): return self.A.mH @ self.A
    def transpose(self): return AtA(self.A)

    def add_diagonal(self, x):
        if isinstance(x, torch.Tensor) and x.numel() <= 1: x = x.item()
        if isinstance(x, (int,float)): x = torch.full((self.shape[0],), fill_value=x, device=self.A.device, dtype=self.A.dtype)
        return Dense(self.to_tensor() + torch.diag_embed(x))

    def solve(self, b):
        *_, n, m = self.A.shape
        if n >= m: return Dense(self.to_tensor()).solve(b)

        A = self.A
        C = A @ A.mH # (n, n), SPD
        L, info = torch.linalg.cholesky_ex(C) # pylint:disable=not-callable
        z = torch.cholesky_solve((A @ b).unsqueeze(-1), L).squeeze(-1)
        return A.mH @ z

    def solve_plus_diag(self, b, diag):
        *_, n, m = self.A.shape
        if (n >= m) or (isinstance(diag, torch.Tensor) and diag.numel() > 1):
            return Dense(self.to_tensor()).solve_plus_diag(b, diag)

        A = self.A
        I = torch.eye(A.size(-2), device=A.device, dtype=A.dtype)

        C = (A @ A.mH).add_(I.mul_(diag)) # (n, n), SPD
        L, info = torch.linalg.cholesky_ex(C + I.mul_(diag)) # pylint:disable=not-callable
        z = torch.cholesky_solve((A @ b).unsqueeze(-1), L).squeeze(-1)
        return (1 / diag) * (b - A.mH @ z)

    def inv(self):
        return Dense(self.to_tensor()).inv()

    def diagonal(self):
        return self.A.pow(2).sum(1)

    def size(self):
        n = self.A.size(1)
        return (n,n)

class AAt(AtA):
    def __init__(self, A: torch.Tensor):
        super().__init__(A.mH)

class Sketched(LinearOperator):
    """A projected by sketching matrix S, representing the operator S @ A_proj @ S.T.

    Where A is (n, n) and S is (n, sketch_size).
    """
    def __init__(self, S: torch.Tensor, A_proj: torch.Tensor):
        self.S = S
        self.A_proj = A_proj
        self.device = self.A_proj.device; self.dtype = self.A_proj.dtype

    def matvec(self, x):
        x_proj = self.S.T @ x
        Ax_proj = self.A_proj @ x_proj
        return self.S @ Ax_proj

    def rmatvec(self, x):
        x_proj = self.S.T @ x
        ATx_proj = self.A_proj.mH @ x_proj
        return self.S @ ATx_proj


    def matmat(self, X): return Dense(torch.linalg.multi_dot([self.S, self.A_proj, self.S.T, X])) # pylint:disable=not-callable
    def rmatmat(self, X): return Dense(torch.linalg.multi_dot([self.S, self.A_proj.mH, self.S.T, X])) # pylint:disable=not-callable


    def is_dense(self): return False
    def to_tensor(self): return self.S @ self.A_proj @ self.S.T
    def transpose(self): return Sketched(self.S, self.A_proj.mH)

    def add_diagonal(self, x):
        """this doesn't correspond to adding diagonal to A, however it still works for LM etc."""
        if isinstance(x, torch.Tensor) and x.numel() <= 1: x = x.item()
        if isinstance(x, (int,float)): x = torch.full((self.A_proj.shape[0],), fill_value=x, device=self.A_proj.device, dtype=self.A_proj.dtype)
        return Sketched(S=self.S, A_proj=self.A_proj + x.diag_embed())

    def solve(self, b):
        return self.S @ torch.linalg.lstsq(self.A_proj, self.S.T @ b).solution # pylint:disable=not-callable

    def inv(self):
        return Sketched(S=self.S, A_proj=torch.linalg.pinv(self.A_proj)) # pylint:disable=not-callable

    def size(self):
        n = self.S.size(0)
        return (n,n)


class Eigendecomposition(LinearOperator):
    """A represented as Q L Q^H. If A is (n,n), then Q is (n, rank); L is a vector - diagonal of (rank, rank)"""
    def __init__(self, L: torch.Tensor, Q: torch.Tensor, use_nystrom: bool = True):
        self.L = L
        self.Q = Q
        self.use_nystrom = use_nystrom
        self.device = self.L.device; self.dtype = self.L.dtype

    def matvec(self, x):
        return self.Q @ ((self.Q.mH @ x) * self.L)

    def rmatvec(self, x):
        return self.matvec(x)

    def matmat(self, X):
        return Dense(self.Q @ (self.L[:, None] * (self.Q.mH @ X)))

    def rmatmat(self, X):
        return self.matmat(X)

    def is_dense(self): return False
    def to_tensor(self): return self.Q @ self.L.diag_embed() @ self.Q.mH
    def transpose(self): return Eigendecomposition(L=self.L, Q=self.Q)

    def add_diagonal(self, x):
        """this doesn't correspond to adding diagonal to A, however it still works for LM etc."""
        if isinstance(x, torch.Tensor) and x.numel() > 1:
            raise RuntimeError("Eigendecomposition linear operator doesn't support add_diagonal with a vector diag")

        return Eigendecomposition(L=self.L + x, Q = self.Q)

    def solve(self, b):
        return self.Q @ ((self.Q.mH @ b) / self.L)

    def solve_plus_diag(self, b, diag):
        if isinstance(diag, torch.Tensor) and diag.numel() > 1: return super().solve_plus_diag(b, diag)
        if not self.use_nystrom: return super().solve_plus_diag(b, diag)
        return nystrom_sketch_and_solve(L=self.L, Q=self.Q, b=b, reg=float(diag))

    def inv(self):
        return Eigendecomposition(L=1 / self.L, Q = self.Q)

    def size(self):
        n = self.Q.size(0)
        return (n,n)

