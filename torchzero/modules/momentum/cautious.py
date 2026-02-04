"""Cautioning related modules"""
from collections import deque
from operator import itemgetter
from typing import Literal

import torch

from ...core import  TensorTransform, Module, Chainable
from ...utils import NumberList, TensorList, unpack_dicts


def cautious_(
    tensors_: TensorList,
    grads: TensorList,
    normalize: bool,
    eps: float,
    mode: Literal['zero', 'grad', 'backtrack']
):
    # mask will be > 0 for parameters where both signs are the same
    mask = (tensors_ * grads) > 0
    if mode in ('zero', 'grad'):
        if normalize and mode == 'zero':
            fmask = mask.to(tensors_[0].dtype)
            fmask /= fmask.global_mean().clip(min=eps) # type:ignore
        else:
            fmask = mask

        tensors_ *= fmask

        if mode == 'grad':
            tensors_ += grads * mask.logical_not_()

        return tensors_

    # mode = 'backtrack'
    tensors_ -= tensors_.mul(2).mul_(mask.logical_not_())
    return tensors_

class Cautious(TensorTransform):
    """Negates update for parameters where update and gradient sign is inconsistent.
    Optionally normalizes the update by the number of parameters that are not masked.
    This is meant to be used after any momentum-based modules.

    Args:
        normalize (bool, optional):
            renormalize update after masking.
            only has effect when mode is 'zero'. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.
            - "zero" - set them to zero (as in paper)
            - "grad" - set them to the gradient (same as using update magnitude and gradient sign)
            - "backtrack" - negate them

    ## Examples:

    Cautious Adam

    ```python
    opt = tz.Optimizer(
        bench.parameters(),
        tz.m.Adam(),
        tz.m.Cautious(),
        tz.m.LR(1e-2)
    )
    ```

    References:
        Cautious Optimizers: Improving Training with One Line of Code. Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu
    """

    def __init__(
        self,
        normalize=False,
        eps=1e-6,
        mode: Literal["zero", "grad", "backtrack"] = "zero",
    ):
        defaults = dict(normalize=normalize, eps=eps, mode=mode)
        super().__init__(defaults, uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        mode, normalize, eps = itemgetter('mode', 'normalize', 'eps')(settings[0])
        return cautious_(TensorList(tensors), TensorList(grads), normalize=normalize, eps=eps, mode=mode)

class UpdateGradientSignConsistency(TensorTransform):
    """Compares update and gradient signs. Output will have 1s where signs match, and 0s where they don't.

    Args:
        normalize (bool, optional):
            renormalize update after masking. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
    """
    def __init__(self, normalize = False, eps=1e-6):

        defaults = dict(normalize=normalize, eps=eps)
        super().__init__(defaults, uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        normalize, eps = itemgetter('normalize', 'eps')(settings[0])

        mask = (TensorList(tensors).mul_(grads)).gt_(0)
        if normalize: mask = mask / mask.global_mean().clip(min = eps) # pyright: ignore[reportOperatorIssue]

        return mask

class IntermoduleCautious(Module):
    """Negaties update on :code:`main` module where it's sign doesn't match with output of ``compare`` module.

    Args:
        main (Chainable): main module or sequence of modules whose update will be cautioned.
        compare (Chainable): modules or sequence of modules to compare the sign to.
        normalize (bool, optional):
            renormalize update after masking. Defaults to False.
        eps (float, optional): epsilon for normalization. Defaults to 1e-6.
        mode (str, optional):
            what to do with updates with inconsistent signs.
            - "zero" - set them to zero (as in paper)
            - "grad" - set them to the gradient (same as using update magnitude and gradient sign)
            - "backtrack" - negate them
    """
    def __init__(
        self,
        main: Chainable,
        compare: Chainable,
        normalize=False,
        eps=1e-6,
        mode: Literal["zero", "grad", "backtrack"] = "zero",
    ):

        defaults = dict(normalize=normalize, eps=eps, mode=mode)
        super().__init__(defaults)

        self.set_child('main', main)
        self.set_child('compare', compare)

    def update(self, objective): raise RuntimeError
    def apply(self, objective): raise RuntimeError

    @torch.no_grad
    def step(self, objective):
        main = self.children['main']
        compare = self.children['compare']

        main_var = main.step(objective.clone(clone_updates=True))
        objective.update_attrs_from_clone_(main_var)

        compare_var = compare.step(objective.clone(clone_updates=True))
        objective.update_attrs_from_clone_(compare_var)

        mode, normalize, eps = itemgetter('mode', 'normalize', 'eps')(self.defaults)
        objective.updates = cautious_(
            TensorList(main_var.get_updates()),
            TensorList(compare_var.get_updates()),
            normalize=normalize,
            mode=mode,
            eps=eps,
        )

        return objective

class ScaleByGradCosineSimilarity(TensorTransform):
    """Multiplies the update by cosine similarity with gradient.
    If cosine similarity is negative, naturally the update will be negated as well.

    Args:
        eps (float, optional): epsilon for division. Defaults to 1e-6.

    ## Examples:

    Scaled Adam
    ```python
    opt = tz.Optimizer(
        bench.parameters(),
        tz.m.Adam(),
        tz.m.ScaleByGradCosineSimilarity(),
        tz.m.LR(1e-2)
    )
    ```
    """
    def __init__(
        self,
        eps: float = 1e-6,
    ):
        defaults = dict(eps=eps)
        super().__init__(defaults, uses_grad=True)

    @torch.no_grad
    def multi_tensor_apply(self, tensors, params, grads, loss, states, settings):
        assert grads is not None
        eps = settings[0]['eps']
        tensors = TensorList(tensors)
        grads = TensorList(grads)
        cos_sim = tensors.dot(grads) / (tensors.global_vector_norm() * grads.global_vector_norm()).clip(min=eps)

        return tensors.mul_(cos_sim)

class ScaleModulesByCosineSimilarity(Module):
    """Scales the output of ``main`` module by it's cosine similarity to the output
    of ``compare`` module.

    Args:
        main (Chainable): main module or sequence of modules whose update will be scaled.
        compare (Chainable): module or sequence of modules to compare to
        eps (float, optional): epsilon for division. Defaults to 1e-6.

    ## Examples:

    Adam scaled by similarity to RMSprop
    ```python
    opt = tz.Optimizer(
        bench.parameters(),
        tz.m.ScaleModulesByCosineSimilarity(
            main = tz.m.Adam(),
            compare = tz.m.RMSprop(0.999, debiased=True),
        ),
        tz.m.LR(1e-2)
    )
    ```
    """
    def __init__(
        self,
        main: Chainable,
        compare: Chainable,
        eps=1e-6,
    ):
        defaults = dict(eps=eps)
        super().__init__(defaults)

        self.set_child('main', main)
        self.set_child('compare', compare)

    def update(self, objective): raise RuntimeError
    def apply(self, objective): raise RuntimeError

    @torch.no_grad
    def step(self, objective):
        main = self.children['main']
        compare = self.children['compare']

        main_var = main.step(objective.clone(clone_updates=True))
        objective.update_attrs_from_clone_(main_var)

        compare_var = compare.step(objective.clone(clone_updates=True))
        objective.update_attrs_from_clone_(compare_var)

        m = TensorList(main_var.get_updates())
        c = TensorList(compare_var.get_updates())
        eps = self.defaults['eps']

        cos_sim = m.dot(c) / (m.global_vector_norm() * c.global_vector_norm()).clip(min=eps)

        objective.updates = m.mul_(cos_sim)
        return objective
