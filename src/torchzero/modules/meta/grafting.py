from collections.abc import Iterable
from typing import Literal
import torch

from ...core import OptimizerModule
from ...tensorlist import TensorList


class Graft(OptimizerModule):
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
        magnitude: OptimizerModule | Iterable[OptimizerModule],
        direction: OptimizerModule | Iterable[OptimizerModule],
        ord: float = 2,
        eps: float = 1e-8,
        layerwise: bool = False,
        # TODO: channelwise
    ):
        super().__init__({})
        self._set_child_('magnitude', magnitude)
        self._set_child_('direction', direction)
        self.ord = ord
        self.eps = eps
        self.layerwise = layerwise


    @torch.no_grad
    def step(self, vars):
        state_copy = vars.copy(clone_ascent=True)
        magnitude = self.children['magnitude'].return_ascent(state_copy)

        if state_copy.grad is not None: vars.grad = state_copy.grad
        if state_copy.fx0 is not None: vars.fx0 = state_copy.fx0
        if state_copy.fx0_approx is not None: vars.fx0_approx = state_copy.fx0_approx

        direction = self.children['direction'].return_ascent(vars)

        if self.layerwise:
            M = magnitude.norm(self.ord)
            D = direction.norm(self.ord)
            D.select_set_(D == 0, M)

        else:
            M = magnitude.total_vector_norm(self.ord)
            D = direction.total_vector_norm(self.ord)
            if D == 0: D = M

        vars.ascent = direction.mul_(M / (D + self.eps))
        return self._update_params_or_step_with_next(vars)



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
        magnitude: OptimizerModule | Iterable[OptimizerModule],
        sign: OptimizerModule | Iterable[OptimizerModule],
    ):
        super().__init__({})

        self._set_child_('magnitude', magnitude)
        self._set_child_('sign', sign)


    @torch.no_grad
    def step(self, vars):
        state_copy = vars.copy(clone_ascent=True)
        magnitude = self.children['magnitude'].return_ascent(state_copy)

        # make sure to store grad and fx0 if it was calculated
        vars.update_attrs_(state_copy)

        sign = self.children['sign'].return_ascent(vars)

        vars.ascent = magnitude.copysign_(sign)
        return self._update_params_or_step_with_next(vars)


class IntermoduleCautious(OptimizerModule):
    """Negates update for parameters where updates of two modules or module chains have inconsistent sign.
    Optionally normalizes the update by the number of parameters that are not masked.

    Args:
        main_module (OptimizerModule | Iterable[OptimizerModule]):
            main module or sequence of modules to chain, which update will be used with a consistency mask applied.
        compare_module (OptimizerModule | Iterable[OptimizerModule]):
            module or sequence of modules to chain, which update will be used to compute a consistency mask.
            Can also be set to `ascent` to compare to update that is passed `main_module`, or `grad` to compare
            to gradients.
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.

            "zero" - set them to zero (as in paper)

            "grad" - set them to the gradient

            "compare_module" - set them to `compare_module`'s update

            "negate" - negate them (same as using update magnitude and gradient sign)
    """
    def __init__(
        self,
        main_module: OptimizerModule | Iterable[OptimizerModule],
        compare_module: OptimizerModule | Iterable[OptimizerModule] | Literal['ascent', 'grad'],
        normalize=False,
        eps=1e-6,
        mode: Literal["zero", "grad", "backtrack", "compare_module"] = "zero",
    ):
        super().__init__({})

        self._set_child_('main',main_module)
        if isinstance(compare_module, str): self.compare_mode = compare_module
        else:
            self._set_child_('compare', compare_module)
            self.compare_mode = 'module'
        self.eps = eps
        self.normalize = normalize
        self.mode: Literal["zero", "grad", "backtrack", "compare_module"]  = mode

    @torch.no_grad
    def step(self, vars):
        params = None
        state_copy = vars.copy(clone_ascent=True)
        ascent = self.children['main'].return_ascent(state_copy)
        vars.update_attrs_(state_copy)

        if self.compare_mode == 'module': compare = self.children['compare'].return_ascent(vars)
        else:
            params = self.get_params()
            if self.compare_mode == 'ascent': compare: TensorList = vars.maybe_use_grad_(params)
            elif self.compare_mode == 'grad': compare: TensorList = vars.maybe_compute_grad_(params)
            else: raise ValueError(f'Invalid compare_module: {self.compare_mode}')

        # mask will be > 0 for parameters where both signs are the same
        mask = (ascent * compare) > 0

        if self.mode == 'backtrack':
            ascent -= ascent.mul(2).mul_(mask.logical_not_())

        else:
            # normalize if mode is `zero`
            if self.normalize and self.mode == 'zero':
                fmask = mask.to(ascent[0].dtype)
                fmask /= fmask.total_mean() + self.eps
            else:
                fmask = mask

            # apply the mask
            ascent *= fmask

            if self.mode == 'grad':
                params = self.get_params()
                ascent += vars.maybe_compute_grad_(params) * mask.logical_not_()

            elif self.mode == 'compare_module':
                ascent += compare * mask.logical_not_()

        vars.ascent = ascent
        return self._update_params_or_step_with_next(vars, params)

