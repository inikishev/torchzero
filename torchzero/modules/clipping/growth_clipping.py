from operator import itemgetter

import torch

from ...core import TensorwiseTransform, Target, Transform
from ...utils import TensorList, as_tensorlist


class ClipValueGrowth(TensorwiseTransform):
    """Clips update value magnitude growth.

    Args:
        add (float | None, optional): additive clipping, next update is at most `previous update + add`. Defaults to None.
        mul (float | None, optional): multiplicative clipping, next update is at most `previous update * mul`. Defaults to 1.5.
        min_value (float | None, optional):
            minimum value for multiplicative clipping to prevent collapse to 0.
            Next update is at most :code:`max(prev_update, min_value) * mul`. Defaults to 1e-4.
        max_decay (float | None, optional):
            bounds the tracked multiplicative clipping decay to prevent collapse to 0.
            Next update is at most :code:`max(previous update * mul, max_decay)`.
            Defaults to 2.
        target (Target, optional): what to set on var. Defaults to "update".
    """
    def __init__(
        self,
        add: float | None = None,
        mul: float | None = 1.5,
        min_value: float | None = 1e-4,
        max_decay: float | None = 2,
        target: Target = "update",
    ):
        defaults = dict(add=add, mul=mul, min_value=min_value, max_decay=max_decay)
        super().__init__(defaults, target=target)


    def apply_tensor(self, tensor, param, grad, loss, state, setting):
        add, mul, min_value, max_decay = itemgetter('add','mul','min_value','max_decay')(setting)
        add: float | None

        if add is None and mul is None:
            return tensor

        if 'prev' not in state:
            state['prev'] = tensor.clone()
            return tensor

        prev: torch.Tensor = state['prev']

        # additive bound
        if add is not None:
            growth = (tensor.abs() - prev.abs()).clip(min=0)
            tensor.sub_(torch.where(growth > add, (growth-add).copysign_(tensor), 0))

        # multiplicative bound
        growth = None
        if mul is not None:
            prev_magn = prev.abs()
            if min_value is not None: prev_magn.clip_(min=min_value)
            growth = (tensor.abs() / prev_magn).clamp_(min=1e-8)

            denom = torch.where(growth > mul, growth/mul, 1)

            tensor.div_(denom)

        # limit max growth decay
        if max_decay is not None:
            if growth is None:
                prev_magn = prev.abs()
                if min_value is not None: prev_magn.clip_(min=min_value)
                growth = (tensor.abs() / prev_magn).clamp_(min=1e-8)

            new_prev = torch.where(growth < (1/max_decay), prev/max_decay, tensor)
        else:
            new_prev = tensor.clone()

        state['prev'] = new_prev
        return tensor


def norm_growth_clip_(
    tensor_: torch.Tensor,
    prev_norm: torch.Tensor,
    add: float | None,
    mul: float | None,
    min_value: float | None,
    max_decay: float | None,
    ord: float,
):
    if add is None and mul is None: return tensor_
    norm = torch.linalg.vector_norm(tensor_, ord=ord) # pylint:disable=not-callable

    denom = 1
    # additive bound
    if add is not None:
        allowed_norm = prev_norm + add
        if norm > allowed_norm: denom = norm / allowed_norm

    # multiplicative bound
    if mul is not None:
        allowed_norm = prev_norm * mul
        if norm > allowed_norm: denom = max(denom, norm / allowed_norm)

    # minimal norm
    if min_value is not None:
        denom = max(denom, min_value)

    # limit max growth decay
    new_prev_norm = norm/denom
    if max_decay is not None:
        decay = norm / prev_norm
        if decay < (1/max_decay):
            new_prev_norm = prev_norm / max_decay

    if min_value is not None: new_prev_norm = max(new_prev_norm, min_value) # pyright:ignore[reportArgumentType]
    return tensor_.div_(denom), new_prev_norm, denom


class ClipNormGrowth(Transform):
    """Clips update norm growth.

    Args:
        add (float | None, optional): additive clipping, next update norm is at most `previous norm + add`. Defaults to None.
        mul (float | None, optional):
            multiplicative clipping, next update norm is at most `previous norm * mul`. Defaults to 1.5.
        min_value (float | None, optional):
            minimum value for multiplicative clipping to prevent collapse to 0.
            Next norm is at most :code:`max(prev_norm, min_value) * mul`. Defaults to 1e-4.
        max_decay (float | None, optional):
            bounds the tracked multiplicative clipping decay to prevent collapse to 0.
            Next norm is at most :code:`max(previous norm * mul, max_decay)`.
            Defaults to 2.
        ord (float, optional): norm order. Defaults to 2.
        parameterwise (bool, optional):
            if True, norms are calculated parameter-wise, otherwise treats all parameters as single vector. Defaults to True.
        target (Target, optional): what to set on var. Defaults to "update".
    """
    def __init__(
        self,
        add: float | None = None,
        mul: float | None = 1.5,
        min_value: float | None = 1e-4,
        max_decay: float | None = 2,
        ord: float = 2,
        parameterwise=True,
        target: Target = "update",
    ):
        defaults = dict(add=add, mul=mul, min_value=min_value, max_decay=max_decay, ord=ord, parameterwise=parameterwise)
        super().__init__(defaults, target=target)



    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        parameterwise = settings[0]['parameterwise']
        tensors = TensorList(tensors)

        if parameterwise:
            ts = tensors
            stts = states
            stns = settings

        else:
            ts = [tensors.to_vec()]
            stts = [self.global_state]
            stns = [settings[0]]


        for t, state, setting in zip(ts, stts, stns):
            if 'prev_norm' not in state:
                state['prev_norm'] = torch.linalg.vector_norm(t, ord=setting['ord']) # pylint:disable=not-callable
                state['prev_denom'] = 1
                continue

            _,  state['prev_norm'], state['prev_denom'] = norm_growth_clip_(
                tensor_ = t,
                prev_norm = state['prev_norm'],
                add = setting['add'],
                mul = setting['mul'],
                min_value = setting['min_value'],
                max_decay = setting['max_decay'],
                ord = setting['ord'],
            )

        if not parameterwise:
            tensors.from_vec_(ts[0])

        return tensors
