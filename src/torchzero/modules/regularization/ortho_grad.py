"""
⟂Grad (read “ortho-grad”) was proposed in https://arxiv.org/abs/2501.04697.

"""
from collections.abc import Iterable

import torch

from ...tensorlist import TensorList
from ...core import OptimizerModule, _Targets


def orthograd_(params: Iterable[torch.Tensor], eps: float = 1e-30):
    """Applies ⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to apply ⟂Grad to.
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)

    reference
        https://arxiv.org/abs/2501.04697
    """
    if not isinstance(params, TensorList): params = TensorList(params)
    params = params.with_grad()
    grad = params.grad
    grad -= (((params*grad).total_sum())/(params*params).total_sum() + eps) * params

class OrthoGrad(OptimizerModule):
    """⟂Grad - projects gradient of an iterable of parameters to be orthogonal to the weights.

    Args:
        params (abc.Iterable[torch.Tensor]): parameters that hold gradients to apply ⟂Grad to.
        eps (float, optional): epsilon added to the denominator for numerical stability (default: 1e-30)
        renormalize (bool, optional): whether to renormalize gradients back to original norm (default: True).
        sqrt_scale (bool, optional):
            uses square root of the scale to make it more impactful, experimental setting and doesn't really work (default: False).
        add (bool, optional):
            Experimental option that changes subtraction to addition.
            I don't think it has any geometric meaning but it drives weights towards zero instead of away from it.
            and it seems to work well with sqrt_scale = True. It speeds up convergence by a lot compared to using vanilla gradient,
            but also has INSANE overfitting.
        target (str, optional):
            determines what this module updates.

            "ascent" - it updates the ascent (default).

            "grad" - it updates the gradient (and sets `.grad` attributes to updated gradient).

            "closure" - it makes a new closure that sets the updated ascent to the .`grad` attributes.

    reference
        https://arxiv.org/abs/2501.04697
    """
    def __init__(self, eps: float = 1e-30, renormalize=True, sqrt_scale = False, add=False, target: _Targets = 'ascent'):
        super().__init__({}, target=target)
        self.eps = eps
        self.add = add
        self.renormalize = renormalize
        self.sqrt_scale = sqrt_scale

    def _update(self, state, ascent):
        params = self.get_params()

        if self.renormalize: orig_norm = ascent.norm(2) + self.eps
        else: orig_norm = 1

        scale = (params*ascent).total_sum() / ((params*params).total_sum() + self.eps)
        if self.sqrt_scale:
            scale = scale.abs().sqrt() * scale.sign()

        if self.add: ascent += params * scale
        else: ascent -= params * scale

        if self.renormalize:
            ascent *= (orig_norm / ascent.norm(2))

        return ascent

