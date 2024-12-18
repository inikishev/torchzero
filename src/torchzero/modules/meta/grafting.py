from collections import abc

import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList
from .chain import Chain
from .return_overrides import ReturnAscent


class Grafting(OptimizerModule):
    """
    Optimizer grafting (magnitude#direction).
    Takes update of one optimizer and makes its norm same as update of another optimizer.
    Can be applied to all weights or layerwise.

    Args:
        magnitude (OptimizerModule | Iterable[OptimizerModule]):
            module to use magnitude from.
            If sequence of modules is provided, they will be chained.
        direction (OptimizerModule | Iterable[OptimizerModule]):
            module/modules to use direction from.
            If sequence of modules is provided, they will be chained.
        ord (int, optional): norm type. Defaults to 2.
        eps (float, optional): epsilon for numerical stability. Defaults to 1e-8.
        layerwise (bool, optional): whether to apply grafting layerwise. Defaults to False.

    reference
        *Agarwal, N., Anil, R., Hazan, E., Koren, T., & Zhang, C.
        Learning Rate Grafting: Transferability of Optimizer Tuning.*
    """
    def __init__(
        self,
        magnitude: OptimizerModule | abc.Iterable[OptimizerModule],
        direction: OptimizerModule | abc.Iterable[OptimizerModule],
        ord: float = 2,
        eps: float = 1e-8,
        layerwise: bool = False,
        # TODO: channelwise
    ):
        super().__init__({})

        if not isinstance(magnitude, abc.Iterable): magnitude = [magnitude]
        if not isinstance(direction, abc.Iterable): direction = [direction]

        self._set_child_('magnitude', Chain([*magnitude, ReturnAscent()]))
        self._set_child_('direction', Chain([*direction, ReturnAscent()]))
        self.ord = ord
        self.eps = eps
        self.layerwise = layerwise


    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent=True)
        magnitude: TensorList = self.children['magnitude'].step(state_copy) # type:ignore

        if state_copy.grad is not None: state.grad = state_copy.grad
        if state_copy.fx0 is not None: state.fx0 = state_copy.fx0
        if state_copy.fx0_approx is not None: state.fx0_approx = state_copy.fx0_approx

        direction: TensorList = self.children['direction'].step(state) # type:ignore

        if self.layerwise:
            M = magnitude.norm(self.ord)
            D = direction.norm(self.ord)
            D.select_set_(D == 0, M)

        else:
            M = magnitude.total_vector_norm(self.ord)
            D = direction.total_vector_norm(self.ord)
            if D == 0: D = M

        state.ascent = direction.mul_(M / (D + self.eps))
        return self._update_params_or_step_with_next(state)



class SignGrafting(OptimizerModule):
    """Weight-wise grafting-like operation where sign of the ascent is taken from first module
    and magnitude from second module.

    Args:
        magnitude (OptimizerModule | Iterable[OptimizerModule]):
            module to take magnitude from.
            If sequence of modules is provided, they will be chained.
        sign (OptimizerModule | Iterable[OptimizerModule]):
            module to take sign from.
            If sequence of modules is provided, they will be chained.
    """
    def __init__(
        self,
        magnitude: OptimizerModule | abc.Iterable[OptimizerModule],
        sign: OptimizerModule | abc.Iterable[OptimizerModule],
    ):
        super().__init__({})

        if not isinstance(magnitude, abc.Iterable): magnitude = [magnitude]
        if not isinstance(sign, abc.Iterable): sign = [sign]

        self._set_child_('magnitude', Chain([*magnitude, ReturnAscent()]))
        self._set_child_('sign', Chain([*sign, ReturnAscent()]))


    @torch.no_grad
    def step(self, state):
        state_copy = state.copy(clone_ascent=True)
        magnitude: TensorList = self.children['magnitude'].step(state_copy).abs_() # type:ignore

        if state_copy.grad is not None: state.grad = state_copy.grad
        if state_copy.fx0 is not None: state.fx0 = state_copy.fx0
        if state_copy.fx0_approx is not None: state.fx0_approx = state_copy.fx0_approx

        sign: TensorList = self.children['sign'].step(state).sign_() # type:ignore

        state.ascent = magnitude.mul_(sign)
        return self._update_params_or_step_with_next(state)

