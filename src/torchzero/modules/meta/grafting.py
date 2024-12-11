from collections import abc

import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList
from .chain import ChainReturn
from .set_grad import ReturnAscent


class Grafting(OptimizerModule):
    def __init__(
        self,
        magnitude: OptimizerModule | abc.Iterable[OptimizerModule],
        direction: OptimizerModule | abc.Iterable[OptimizerModule],
        ord = 2,
        eps = 1e-8,
        layerwise=False,
    ):
        """
        Optimizer grafting (magnitude#direction).
        Takes update of one optimizer and makes its norm same as update of another optimizer.
        Can be applied to all weights or layerwise.

        *Agarwal, N., Anil, R., Hazan, E., Koren, T., & Zhang, C. Learning Rate Grafting: Transferability of Optimizer Tuning.*

        Args:
            magnitude (OptimizerModule | abc.Iterable[OptimizerModule]): modules to use magnitude from.
            direction (OptimizerModule | abc.Iterable[OptimizerModule]): modules to use direction from.
            ord (int, optional): norm type. Defaults to 2.
            eps (_type_, optional): epsilon for numerical stability. Defaults to 1e-8.
            layerwise (bool, optional): whether to apply grafting layerwise. Defaults to False.
        """
        super().__init__({})

        if not isinstance(magnitude, abc.Iterable): magnitude = [magnitude]
        if not isinstance(direction, abc.Iterable): direction = [direction]

        self._add_child_(ChainReturn([*magnitude, ReturnAscent()]))
        self._add_child_(ChainReturn([*direction, ReturnAscent()]))
        self.ord = ord
        self.eps = eps
        self.layerwise = layerwise


    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent=True)
        magnitude: TensorList = self.children[0].step(state_copy) # type:ignore

        if state_copy.grad is not None: state.grad = state_copy.grad
        if state_copy.fx0 is not None: state.fx0 = state_copy.fx0
        if state_copy.fx0_approx is not None: state.fx0_approx = state_copy.fx0_approx

        direction: TensorList = self.children[1].step(state) # type:ignore

        if self.layerwise:
            M = magnitude.abs_().pow_(self.ord).sum().pow_(1/self.ord)
            D = direction.abs().pow_(self.ord).sum().pow_(1/self.ord)
            D.select_set_(D == 0, M)

        else:
            M = magnitude.total_vector_norm(self.ord)
            D = direction.total_vector_norm(self.ord)
            if D == 0: D = M

        state.ascent = direction.mul_(M / (D + self.eps))
        return self._update_params_or_step_with_next(state)



class SignGrafting(OptimizerModule):
    def __init__(
        self,
        magnitude: OptimizerModule | abc.Iterable[OptimizerModule],
        sign: OptimizerModule | abc.Iterable[OptimizerModule],
    ):
        """Weight-wise grafting-like operation where sign of the ascent is taken from first module
        and magnitude from second module.

        Args:
            magnitude (OptimizerModule | abc.Iterable[OptimizerModule]): modules to take magnitude from.
            sign (OptimizerModule | abc.Iterable[OptimizerModule]): modules to take sign from.
        """
        super().__init__({})

        if not isinstance(magnitude, abc.Iterable): magnitude = [magnitude]
        if not isinstance(sign, abc.Iterable): sign = [sign]

        self._add_child_(ChainReturn([*magnitude, ReturnAscent()]))
        self._add_child_(ChainReturn([*sign, ReturnAscent()]))


    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent=True)
        magnitude: TensorList = self.children[0].step(state_copy).abs_() # type:ignore

        if state_copy.grad is not None: state.grad = state_copy.grad
        if state_copy.fx0 is not None: state.fx0 = state_copy.fx0
        if state_copy.fx0_approx is not None: state.fx0_approx = state_copy.fx0_approx

        sign: TensorList = self.children[1].step(state).sign_() # type:ignore

        state.ascent = magnitude.mul_(sign)
        return self._update_params_or_step_with_next(state)

