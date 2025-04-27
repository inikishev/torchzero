from collections.abc import Sequence
from operator import itemgetter

import numpy as np
import torch

from ...core import Chainable, Transform, apply
from ...linalg import matrix_power_eig, matrix_power_svd
from ...utils import set_storage_


def update_shampoo_preconditioner_(
    grad: torch.Tensor,
    accumulators_: list[torch.Tensor | None],
    preconditioners_: list[torch.Tensor | None],
    step: int,
    update_freq: int,
    exp_override: int | None,
    beta: float | None,
    matrix_eps: float,
):
    for i, (accumulator, preconditioner) in enumerate(zip(accumulators_, preconditioners_)):
        if accumulator is None: continue
        assert preconditioner is not None

        axes = list(range(i)) + list(range(i + 1, grad.ndim))
        if beta is None: accumulator.add_(torch.tensordot(grad, grad, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: accumulator.lerp_(torch.tensordot(grad, grad, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

        if step % update_freq == 0:
            matrix_exp = -1/(grad.ndim*2) if exp_override is None else -1/exp_override
            #matrix_exp = exp if exp is not None else grad.ndim*2
            set_storage_(preconditioner, matrix_power_eig(accumulator, matrix_exp, matrix_eps))


def apply_shampoo_preconditioner(
    tensor: torch.Tensor,
    preconditioners_: list[torch.Tensor | None],
    decay: float | None,
):
    for i, preconditioner in enumerate(preconditioners_):
        if preconditioner is None: continue
        tensor = torch.tensordot(tensor, preconditioner, ([0], [0])) # pyright:ignore[reportArgumentType]
        if decay is not None: preconditioner.mul_(decay)
    return tensor


def update_diagonal_(grad: torch.Tensor, diagonal_accumulator_: torch.Tensor, beta: float | None):
    if beta is None: diagonal_accumulator_.add_(grad)
    else: diagonal_accumulator_.lerp_(grad, 1-beta)

def apply_diagonal_(grad_: torch.Tensor, diagonal_accumulator_: torch.Tensor, decay: float | None, eps: float):
    grad_.div_(diagonal_accumulator_ + eps)
    if decay is not None: diagonal_accumulator_.mul_(decay)
    return grad_

def _merge_small_dims(tensor: torch.Tensor, max_dim: int):
    """a safer merger"""
    if tensor.ndim == 0: return tensor, None, None
    sort_idxs = np.argsort(tensor.shape)
    if tensor.shape[sort_idxs[0]] > max_dim:
        return tensor, None, None

    tensor = tensor.permute(*sort_idxs)
    flatten_end_idx = 0
    flat_sizes = []
    flat_numel = 1
    for i, size in enumerate(tensor.shape):
        if flat_numel * size <= max_dim:
            flatten_end_idx = i
            flat_numel *= size
            flat_sizes.append(size)
        else:
            break

    if flatten_end_idx != 0:
        tensor = tensor.flatten(end_dim=flatten_end_idx)

    return tensor, flat_sizes, sort_idxs

def _unmerge_small_dims(tensor: torch.Tensor, flat_sizes: Sequence[int] | None, sort_idxs: np.ndarray | Sequence[int] | None):
    if flat_sizes is None: return tensor
    assert sort_idxs is not None
    tensor = tensor.unflatten(0, flat_sizes)
    return tensor.permute(*np.argsort(sort_idxs))


class Shampoo(Transform):
    """Shampoo: Preconditioned Stochastic Tensor Optimization (https://arxiv.org/abs/1802.09568).

    Args:
        decay (float | None, optional): slowly decays preconditioners. Defaults to None.
        beta (float | None, optional): lerps preconditioners, if None calculates sum as in standard shampoo. Defaults to None.
        matrix_eps (float, optional): epsilon for matrix operations. Defaults to 1e-10.
        update_freq (int, optional): preconditioner update frequence. Defaults to 10.
        exp (int | None, optional): matrix exponent. Defaults to None.
        inner (Chainable | None, optional):
            module applied after updating preconditioners and before applying preconditioning.
            For example if beta≈0.999 and `inner=tz.m.EMA(0.9)`, this becomes Adam with shampoo preconditioner (ignoring debiasing).
            Defaults to None.
    """
    def __init__(
        self,
        decay: float | None = None,
        beta: float | None = None,
        matrix_eps: float = 1e-6,
        update_freq: int = 10,
        exp_override: int | None = None,
        merge_small: bool = True,
        max_dim: int = 2_000,
        precondition_1d=True,
        adagrad_eps=1e-8,
        inner: Chainable | None = None,
    ):
        defaults = dict(decay=decay, beta=beta, matrix_eps=matrix_eps, update_freq=update_freq, exp_override=exp_override, merge_small=merge_small, max_dim=max_dim, precondition_1d=precondition_1d,adagrad_eps=adagrad_eps)
        super().__init__(defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    def transform(self, target, params, grad, vars):
        merged_target = [] # target with merged dims

        # update preconditioners
        for i,(p,t) in enumerate(zip(params, target)):
            state = self.state[p]
            settings = self.settings[p]
            beta, matrix_eps, update_freq, exp_override, merge_small, max_dim, precondition_1d = itemgetter(
                'beta', 'matrix_eps', 'update_freq', 'exp_override', 'merge_small', 'max_dim', 'precondition_1d')(settings)

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)
            merged_target.append(t)

            # initialize accumulators and preconditioners for each dim on 1st step
            if 'accumulators' not in state:

                if not precondition_1d and t.ndim <= 1:
                    state['accumulators'] = []

                else:
                    state['accumulators'] = [matrix_eps * torch.eye(s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]
                    state['preconditioners'] = [torch.eye(s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]

                # either scalar parameter, 1d with precondition_1d=False, or too big, then basic diagonal preconditioner is used.
                if len([i is not None for i in state['accumulators']]) == 0:
                    state['diagonal_accumulator'] = torch.zeros_like(t)

                state['step'] = 0

            # update preconditioners
            if 'diagonal_accumulator' in state:
                update_diagonal_(t, state['diagonal_accumulator'], beta)
            else:
                update_shampoo_preconditioner_(
                    t,
                    accumulators_=state['accumulators'],
                    preconditioners_=state['preconditioners'],
                    step=state['step'],
                    update_freq=update_freq,
                    exp_override=exp_override,
                    beta=beta,
                    matrix_eps=matrix_eps,
                )

        # inner step
        if 'inner' in self.children:
            target = apply(self.children['inner'], target, params=params, grad=grad, vars=vars)

            # have to merge small dims again
            merged_target = [] # target with merged dims
            for i,(p,t) in enumerate(zip(params, target)):
                state = self.state[p]
                settings = self.settings[p]
                if settings['merge_small']:
                    t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, settings['max_dim'])
                merged_target.append(t)

        # precondition
        for i, (p, t) in enumerate(zip(params, merged_target)):
            state = self.state[p]
            settings = self.settings[p]
            decay, merge_small, adagrad_eps= itemgetter('decay', 'merge_small', 'adagrad_eps')(settings)

            if 'diagonal_accumulator' in state:
                target[i] = apply_diagonal_(t, state['diagonal_accumulator'], decay=decay, eps=adagrad_eps)
            else:
                target[i] = apply_shampoo_preconditioner(t, preconditioners_=state['preconditioners'], decay=decay)

            if merge_small:
                target[i] = _unmerge_small_dims(target[i], state['flat_sizes'], state['sort_idxs'])

            state['step'] += 1

        return target