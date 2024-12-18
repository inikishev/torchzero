import torch

from ...core import TensorListOptimizer, ClosureType


class RandomSearch(TensorListOptimizer):
    """Pure random search.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        min (float, optional): lower bound of the search space. Defaults to -10.
        max (float, optional): upper bound of the search space. Defaults to 10.
        stochastic (bool, optional):
            evaluate function twice per step,
            and only accept new params if they decreased the loss.
            Defaults to False.
    """
    def __init__(self, params, min:float = -10, max:float = 10, stochastic = False):
        defaults = dict(min=min, max = max)
        super().__init__(params, defaults)
        self.lowest_loss = float('inf')
        self.stochastic = stochastic

    @torch.no_grad
    def step(self, closure: ClosureType): # type:ignore # pylint:disable=W0222
        if self.stochastic: self.lowest_loss = closure()

        settings = self.get_all_group_keys()
        params = self.get_params()

        old_params = params.clone()
        new_params = params.uniform_like(settings['min'], settings['max'])
        params.set_(new_params)
        loss = closure(False)

        if loss < self.lowest_loss: self.lowest_loss = loss
        else: params.set_(old_params)
        return loss

class CyclicRS(TensorListOptimizer):
    """Performs random search cycling through each coordinate.
    Works surprisingly well on up to ~100 dimensional problems.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        min (float, optional): lower bound of the search space. Defaults to -10.
        max (float, optional): upper bound of the search space. Defaults to 10.
        stochastic (bool, optional):
            evaluate function twice per step,
            and only accept new params if they decreased the loss.
            Defaults to False.
    """
    def __init__(self, params, min:float = -10, max:float = 10, stochastic = False):
        defaults = dict(min=min, max = max)
        super().__init__(params, defaults)
        self.lowest_loss = float('inf')
        self.cur_param = 0
        self.cur_value = 0
        self.stochastic = stochastic

    @torch.no_grad
    def step(self, closure: ClosureType): # type:ignore # pylint:disable=W0222
        if self.stochastic: self.lowest_loss = closure()
        settings = self.get_all_group_keys()
        params = self.get_params()

        if self.cur_param >= len(params): self.cur_param = 0
        param = params[self.cur_param]
        if self.cur_value >= param.numel():
            self.cur_value = 0
            self.cur_param += 1
            if self.cur_param >= len(params): self.cur_param = 0
            param = params[self.cur_param]

        flat = param.view(-1)
        old_value = flat[self.cur_value].clone()
        flat[self.cur_value] = torch.rand(1).uniform_(settings['min'][self.cur_param], settings['max'][self.cur_param]) # type:ignore

        loss = closure(False)
        if loss < self.lowest_loss: self.lowest_loss = loss
        else:
            flat[self.cur_value] = old_value

        self.cur_value += 1
        return loss
