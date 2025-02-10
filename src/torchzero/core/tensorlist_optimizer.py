from typing import Literal, Any, overload, TypeVar
from abc import ABC
from collections.abc import Callable, Sequence, Iterable, Mapping, MutableSequence
import numpy as np
import torch
import torch.optim.optimizer
from torch.optim.optimizer import ParamsT

from ..tensorlist import TensorList, NumberList
from ..utils.torch_tools import totensor, tofloat
from ..utils.python_tools import _ScalarLoss

_StateInit = Literal['params', 'grad'] | Callable | TensorList

_ClosureType = Callable[..., _ScalarLoss]
"""

Closure example:

.. code-block:: python

    def closure(backward = True):
        loss = model(inputs)
        if backward:
            optimizer.zero_grad()
            loss.backward()
        return loss

This closure will also work with all built in pytorch optimizers including LBFGS, as well as and most custom ones.
"""

def _maybe_pass_backward(closure: _ClosureType, backward: bool) -> _ScalarLoss:
    """not passing backward when it is true makes this work with closures with no `backward` argument"""
    if backward:
        with torch.enable_grad(): return closure()
    return closure(False)

CLS = TypeVar('CLS')
class TensorListOptimizer(torch.optim.Optimizer, ABC):
    """torch.optim.Optimizer with some additional methods related to TensorList.

    Args:
        params (ParamsT): iterable of parameters.
        defaults (_type_): dictionary with default parameters for the optimizer.
    """
    def __init__(self, params: ParamsT, defaults):
        super().__init__(params, defaults)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]
        self.has_complex = any(torch.is_complex(x) for x in self._params)
        """True if any of the params are complex"""

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]
        self.has_complex = any(torch.is_complex(x) for x in self._params)

    # def get_params[CLS: Any](self, cls: type[CLS] = TensorList) -> CLS:
    def get_params(self, cls: type[CLS] = TensorList) -> CLS:
        """returns all params with `requires_grad = True` as a TensorList."""
        return cls(p for p in self._params if p.requires_grad) # type:ignore

    def ensure_grad_(self):
        """Replaces None grad attribute with zeroes for all parameters that require grad."""
        for p in self.get_params():
            if p.requires_grad and p.grad is None: p.grad = torch.zeros_like(p)

    # def get_state_key[CLS: MutableSequence](self, key: str, init: _StateInit = torch.zeros_like, params=None, cls: type[CLS] = TensorList) -> CLS:
    def get_state_key(self, key: str, init: _StateInit = torch.zeros_like, params=None, cls: type[CLS] = TensorList) -> CLS:
        """Returns a tensorlist of all `key` states of all params with `requires_grad = True`.

        Args:
            key (str): key to create/access.
            init: Initial value if key doesn't exist. Can be `params`, `grad`, or callable such as `torch.zeros_like`.
                Defaults to torch.zeros_like.
            params (optional): optionally pass params if you already created them. Defaults to None.
            cls (optional): optionally specify any other MutableSequence subclass to use instead of TensorList.

        Returns:
            TensorList: TensorList with the `key` state. Those tensors are stored in the optimizer, so modify them in-place.
        """
        value = cls()
        if params is None: params = self.get_params()
        for pi, p in enumerate(params):
            state = self.state[p]
            if key not in state:
                if callable(init): state[key] = init(p)
                elif isinstance(init, TensorList): state[key] = init[pi].clone()
                elif init == 'params': state[key] = p.clone().detach()
                elif init == 'grad': state[key] = p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                else: raise ValueError(f'unknown init - {init}')
            value.append(state[key]) # type:ignore
        return value

    # def get_state_keys[CLS: MutableSequence](
    def get_state_keys(
        self,
        *keys: str,
        inits: _StateInit | Sequence[_StateInit] = torch.zeros_like,
        params=None,
        cls: type[CLS] = TensorList,
    ) -> list[CLS]:
        """Returns a TensorList with the `key` states of all `params`. Creates the states if they don't exist."""

        values = [cls() for _ in range(len(keys))]
        if params is None: params = self.get_params()
        if callable(inits) or isinstance(inits, str): inits = [inits] * len(keys) # type:ignore

        for pi, p in enumerate(params):
            state = self.state[p]
            for i, (key, init) in enumerate(zip(keys, inits)): # type:ignore
                if key not in state:
                    if callable(init): state[key] = init(p)
                    elif isinstance(init, TensorList): state[key] = init[pi].clone()
                    elif init == 'params': state[key] = p.clone().detach()
                    elif init == 'grad': state[key] = p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                    else: raise ValueError(f'unknown init - {init}')
                values[i].append(state[key]) # type:ignore
        return values

    def _yield_groups_key(self, key: str):
        for group in self.param_groups:
            value = group[key]
            for p in group['params']:
                if p.requires_grad: yield value


    # def get_group_key[CLS: Any](self, key: str, cls: type[CLS] = NumberList) -> CLS:
    def get_group_key(self, key: str, cls: type[CLS] = NumberList) -> CLS:
        """Returns a TensorList with the param_groups `key` setting of each param."""
        return cls(self._yield_groups_key(key)) # type:ignore

    def get_first_group_key(self, key:str) -> Any:
        """Returns the param_groups `key` setting of the first param."""
        return next(iter(self._yield_groups_key(key)))

    # def get_all_group_keys[CLS: Any](self, cls: type[CLS] = NumberList) -> dict[str, CLS]:
    def get_all_group_keys(self, cls: type[CLS] = NumberList) -> dict[str, CLS]:
        all_values: dict[str, CLS] = {}
        for group in self.param_groups:

            n_params = len([p for p in group['params'] if p.requires_grad])

            for key, value in group.items():
                if key != 'params':
                    if key not in all_values: all_values[key] = cls(value for _ in range(n_params)) # type:ignore
                    else: all_values[key].extend([value for _ in range(n_params)]) # type:ignore

        return all_values

    # def get_group_keys[CLS: MutableSequence](self, *keys: str, cls: type[CLS] = NumberList) -> list[CLS]:
    def get_group_keys(self, *keys: str, cls: type[CLS] = NumberList) -> list[CLS]:
        """Returns a list with the param_groups `key` setting of each param."""

        all_values: list[CLS] = [cls() for _ in keys]
        for group in self.param_groups:

            n_params = len([p for p in group['params'] if p.requires_grad])

            for i, key in enumerate(keys):
                value = group[key]
                all_values[i].extend([value for _ in range(n_params)]) # type:ignore

        return all_values

    @torch.no_grad
    def evaluate_loss_at_vec(self, vec, closure=None, params = None, backward=False, ensure_float=False):
        """_summary_

        Args:
            vec (_type_): _description_
            closure (_type_, optional): _description_. Defaults to None.
            params (_type_, optional): _description_. Defaults to None.
            backward (bool, optional): _description_. Defaults to False.
            ensure_float (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        vec = totensor(vec)
        if closure is None: closure = self._closure # type:ignore # pylint:disable=no-member
        if params is None: params = self.get_params()

        params.from_vec_(vec.to(params[0]))
        loss = _maybe_pass_backward(closure, backward)

        if ensure_float: return tofloat(loss)
        return _maybe_pass_backward(closure, backward)

    @overload
    def evaluate_loss_grad_at_vec(self, vec, closure=None, params = None, to_numpy: Literal[True] = False) -> tuple[float, np.ndarray]: ... # type:ignore
    @overload
    def evaluate_loss_grad_at_vec(self, vec, closure=None, params = None, to_numpy: Literal[False] = False) -> tuple[_ScalarLoss, torch.Tensor]: ...
    @torch.no_grad
    def evaluate_loss_grad_at_vec(self, vec, closure=None, params = None, to_numpy: Literal[True] | Literal[False]=False):
        """_summary_

        Args:
            vec (_type_): _description_
            closure (_type_, optional): _description_. Defaults to None.
            params (_type_, optional): _description_. Defaults to None.
            to_numpy (Literal[True] | Literal[False], optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if params is None: params = self.get_params()
        loss = self.evaluate_loss_at_vec(vec, closure, params, backward = True, ensure_float = to_numpy)
        grad = params.grad.to_vec()

        if to_numpy: return tofloat(loss), grad.detach().cpu().numpy()
        return loss, grad


    @torch.no_grad
    def _maybe_evaluate_closure(self, closure, backward=True):
        loss = None
        if closure is not None:
            loss = _maybe_pass_backward(closure, backward)
        return loss