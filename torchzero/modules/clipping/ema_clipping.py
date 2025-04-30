from operator import itemgetter
from typing import Literal
from collections.abc import Iterable, Sequence

import torch

from ...core import Module, Target, Transform, apply, Chainable
from ...utils import NumberList, TensorList, generic_eq

class ClipNormByEMA(Transform):
    NORMALIZE = False
    def __init__(
        self,
        beta=0.99,
        ord: float = 2,
        eps=1e-6,
        tensorwise:bool=True,
        max_ema_growth: float | None = 1.5,
        ema_init: Literal['zeros', 'update'] = 'zeros',
        target: Target = "update",
    ):
        defaults = dict(beta=beta, ord=ord, tensorwise=tensorwise, ema_init=ema_init, eps=eps, max_ema_growth=max_ema_growth)
        super().__init__(defaults, uses_grad=False, target=target)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        ord, tensorwise, ema_init, max_ema_growth = itemgetter('ord', 'tensorwise', 'ema_init', 'max_ema_growth')(self.settings[params[0]])

        beta, eps = self.get_settings('beta', 'eps', params=params, cls=NumberList)
        target = TensorList(target)

        ema = self.get_state('ema', params=params, init = (torch.zeros_like if ema_init=='zeros' else target), cls=TensorList)
        ema.lerp_(target, 1-beta)

        if tensorwise:
            ema_norm = ema.norm(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = self.get_state('prev_ema_norm', params=params, init=ema_norm, cls=TensorList)
                allowed_norm = (prev_ema_norm * max_ema_growth).clip(min=1e-6)
                ema_denom = (ema_norm / allowed_norm).clip(min=1)
                ema.div_(ema_denom)
                ema_norm.div_(ema_denom)
                prev_ema_norm.set_(ema_norm)

            target_norm = target.norm(ord)
            denom = target_norm / ema_norm.clip(min=eps)
            if self.NORMALIZE: denom.clip_(min=eps)
            else: denom.clip_(min=1)

        else:
            ema_norm = ema.global_vector_norm(ord)

            # clip ema norm growth
            if max_ema_growth is not None:
                prev_ema_norm = self.global_state.setdefault('prev_ema_norm', ema_norm)
                allowed_norm = prev_ema_norm * max_ema_growth
                if ema_norm > allowed_norm:
                    ema.div_(ema_norm / allowed_norm)
                    ema_norm = allowed_norm
                prev_ema_norm.set_(ema_norm)

            target_norm = target.global_vector_norm(ord)
            denom = target_norm / ema_norm.clip(min=eps[0])
            if self.NORMALIZE: denom.clip_(min=eps[0])
            else: denom.clip_(min=1)

        target.div_(denom)
        return target

class NormalizeByEMA(ClipNormByEMA):
    NORMALIZE = True



class ClipValueByEMA(Transform):
    def __init__(
        self,
        beta=0.99,
        ema_init: Literal['zeros', 'update'] = 'zeros',
        target: Target = "update",
        ema_tfm:Chainable | None=None,
    ):
        defaults = dict(beta=beta, ema_init=ema_init)
        super().__init__(defaults, uses_grad=False, target=target)

        if ema_tfm is not None:
            self.set_child('ema_tfm', ema_tfm)

    @torch.no_grad
    def transform(self, target, params, grad, vars):
        ema_init = itemgetter('ema_init')(self.settings[params[0]])

        beta = self.get_settings('beta', params=params, cls=NumberList)
        target = TensorList(target)

        ema = self.get_state('ema', params=params, init = (torch.zeros_like if ema_init=='zeros' else lambda t: t.abs()), cls=TensorList)
        ema.lerp_(target.abs(), 1-beta)

        if 'ema_tfm' in self.children:
            ema = TensorList(apply(self.children['ema_tfm'], ema, params, vars.grad, vars))

        target.clip_(-ema, ema)
        return target
