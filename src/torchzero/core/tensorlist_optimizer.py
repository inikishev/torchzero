import typing as T
from abc import ABC
from collections import abc

import torch
import torch.optim.optimizer
from torch.optim.optimizer import ParamsT

from torchzero.tensorlist import TensorList, NumberList

_StateInit = T.Literal['params', 'grad'] | abc.Callable
class TensorListOptimizer(torch.optim.Optimizer, ABC):
    """torch.optim.Optimizer with some additional methods."""
    def __init__(self, params: ParamsT, defaults):
        super().__init__(params, defaults)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]

    def add_param_group(self, param_group: dict[str, T.Any]) -> None:
        super().add_param_group(param_group)
        self._params: list[torch.Tensor] = [param for group in self.param_groups for param in group['params']]

    def get_params(self) -> TensorList:
        return TensorList(p for p in self._params if p.requires_grad)

    def create_zero_grad_(self):
        """Replaces None grad attribute with zeroes for all parameters that require grad."""
        for p in self.get_params():
            if p.requires_grad and p.grad is None: p.grad = torch.zeros_like(p)

    def get_state_key(self, key: str, init: T.Literal['params', 'grad'] | abc.Callable = torch.zeros_like) -> TensorList:
        """Returns a TensorList with the `key` states of all `params`. Creates the states if they don't exist."""
        value = TensorList()
        params = self.get_params()
        for p in params:
            state = self.state[p]
            if key not in state:
                if callable(init): state[key] = init(p)
                elif init == 'params': state[key] = params.clone()
                elif init == 'grad': state[key] = params.grad.clone()
                else: raise ValueError(f'unknown init - {init}')
            value.append(state[key])
        return value

    def get_state_keys(self, keys: abc.Sequence[str], inits: _StateInit | abc.Sequence[_StateInit] = torch.zeros_like) -> list[TensorList]:
        """Returns a TensorList with the `key` states of all `params`. Creates the states if they don't exist."""
        if isinstance(keys, str): raise TypeError('keys must be a sequence of strings')

        values = [TensorList() for _ in range(len(keys))]
        params = self.get_params()
        if callable(inits) or isinstance(inits, str): inits = [inits] * len(keys) # type:ignore

        for p in params:
            state = self.state[p]
            for i, (key, init) in enumerate(zip(keys, inits)): # type:ignore
                if key not in state:
                    if callable(init): state[key] = init(p)
                    elif init == 'params': state[key] = params.clone()
                    elif init == 'grad': state[key] = params.grad.clone()
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

            n_params = len([i for i in group['params'] if i.requires_grad])

            for key, value in group.items():
                if key != 'params':
                    if key not in all_values: all_values[key] = cls(value for _ in range(n_params))
                    else: all_values[key].extend([value for _ in range(n_params)])

        return all_values
