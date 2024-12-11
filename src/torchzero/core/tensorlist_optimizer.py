import typing as T
from abc import ABC
from collections import abc

import torch
import torch.optim.optimizer
from torch.optim.optimizer import ParamsT

from torchzero.tensorlist import TensorList, NumberList

_StateInit = T.Literal['params', 'grad'] | abc.Callable | TensorList
class TensorListOptimizer(torch.optim.Optimizer, ABC):
    def __init__(self, params: ParamsT, defaults):
        """torch.optim.Optimizer with some additional methods related to TensorList.

        Args:
            params (ParamsT): iterable of parameters.
            defaults (_type_): dictionary with default parameters for the optimizer.
        """
        super().__init__(params, defaults)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]

    def add_param_group(self, param_group: dict[str, T.Any]) -> None:
        super().add_param_group(param_group)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]


    def get_params(self) -> TensorList:
        return TensorList(p for p in self._params if p.requires_grad)

    def ensure_grad_(self):
        """Replaces None grad attribute with zeroes for all parameters that require grad."""
        for p in self.get_params():
            if p.requires_grad and p.grad is None: p.grad = torch.zeros_like(p)

    def get_state_key(self, key: str, init: _StateInit = torch.zeros_like, params=None) -> TensorList:
        """Returns a TensorList with the `key` states of all `params` that currently have grad or require grad,
        depending on `mode` passed on `__init__`. Creates the states if they don't exist.
        This guarantees that the returned TensorList is the same shape as params, so
        models where some parameters don't have a gradient sometimes will still work.

        Args:
            key (str): key to create/access.
            init: Initial value if key doesn't exist. Can be `params`, `grad`, or callable such as `torch.zeros_like`.
                Defaults to torch.zeros_like.
            params (_type_, optional): optionally pass params if you already created them. Defaults to None.

        Returns:
            TensorList: TensorList with the `key` state. Those tensors are stored in the optimizer, so modify them in-place.
        """
        value = TensorList()
        if params is None: params = self.get_params()
        for pi, p in enumerate(params):
            state = self.state[p]
            if key not in state:
                if callable(init): state[key] = init(p)
                elif isinstance(init, TensorList): state[key] = init[pi]
                elif init == 'params': state[key] = params.clone()
                elif init == 'grad': state[key] = params.grad.clone()
                else: raise ValueError(f'unknown init - {init}')
            value.append(state[key])
        return value

    def get_state_keys(
        self,
        keys: abc.Sequence[str],
        inits: _StateInit | abc.Sequence[_StateInit] = torch.zeros_like,
        params=None,
    ) -> list[TensorList]:
        """Returns a TensorList with the `key` states of all `params`. Creates the states if they don't exist."""
        if isinstance(keys, str): raise TypeError('keys must be a sequence of strings')

        values = [TensorList() for _ in range(len(keys))]
        if params is None: params = self.get_params()
        if callable(inits) or isinstance(inits, str): inits = [inits] * len(keys) # type:ignore

        for pi, p in enumerate(params):
            state = self.state[p]
            for i, (key, init) in enumerate(zip(keys, inits)): # type:ignore
                if key not in state:
                    if callable(init): state[key] = init(p)
                    elif isinstance(init, TensorList): state[key] = init[pi]
                    elif init == 'params': state[key] = p.clone()
                    elif init == 'grad': state[key] = p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                    else: raise ValueError(f'unknown init - {init}')
                values[i].append(state[key])
        return values

    def _yield_groups_key(self, key: str):
        for group in self.param_groups:
            value = group[key]
            for p in group['params']:
                if p.requires_grad: yield value


    def get_group_key(self, key: str, cls = NumberList) -> TensorList:
        """Returns a TensorList with the param_groups `key` setting of each param."""
        return cls(self._yield_groups_key(key))

    def get_first_group_key(self, key:str) -> T.Any:
        """Returns the param_groups `key` setting of the first param."""
        return next(iter(self._yield_groups_key(key)))

    def get_all_group_keys(self, cls = NumberList) -> dict[str, TensorList]:
        all_values: dict[str, TensorList] = {}
        for group in self.param_groups:

            n_params = len([p for p in group['params'] if p.requires_grad])

            for key, value in group.items():
                if key != 'params':
                    if key not in all_values: all_values[key] = cls(value for _ in range(n_params))
                    else: all_values[key].extend([value for _ in range(n_params)])

        return all_values

    def get_group_keys(self, keys: abc.Sequence[str], cls = NumberList) -> list[TensorList]:
        """Returns a TensorList with the param_groups `key` setting of each param."""
        if isinstance(keys, str): raise TypeError('keys must be a sequence of strings')

        all_values: list[TensorList] = [cls() for _ in keys]
        for group in self.param_groups:

            n_params = len([p for p in group['params'] if p.requires_grad])

            for i, key in enumerate(keys):
                value = group[key]
                all_values[i].extend([value for _ in range(n_params)])

        return all_values