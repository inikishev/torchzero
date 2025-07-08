from collections.abc import Sequence
from operator import itemgetter
from functools import partial
import numpy as np
import torch

from ...core import Chainable, Transform, apply_transform
from ...utils.linalg import matrix_power_eigh
from ...utils import set_storage_


def update_shampoo_preconditioner_(
    grad: torch.Tensor,
    accumulators_: list[torch.Tensor | None],
    preconditioners_: list[torch.Tensor | None],
    step: int,
    update_freq: int,
    exp_override: int | None,
    beta: float | None,
):
    for i, (accumulator, preconditioner) in enumerate(zip(accumulators_, preconditioners_)):
        if accumulator is None: continue
        assert preconditioner is not None

        axes = list(range(i)) + list(range(i + 1, grad.ndim))
        if beta is None: accumulator.add_(torch.tensordot(grad, grad, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: accumulator.lerp_(torch.tensordot(grad, grad, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

        if step % update_freq == 0:
            matrix_exp = -1/(grad.ndim*2) if exp_override is None else -1/exp_override
            set_storage_(preconditioner, matrix_power_eigh(accumulator, matrix_exp))


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
    if beta is None: diagonal_accumulator_.add_(grad.pow(2))
    else: diagonal_accumulator_.mul_(beta).addcmul_(grad, grad, value=1-beta)

def apply_diagonal_(grad_: torch.Tensor, diagonal_accumulator_: torch.Tensor, decay: float | None, eps: float):
    grad_.div_(diagonal_accumulator_.sqrt() + eps)
    if decay is not None: diagonal_accumulator_.mul_(decay)
    return grad_

def _merge_small_dims(tensor: torch.Tensor, max_dim: int):
    """a safer merger"""
    if tensor.ndim == 0: return tensor, None, None
    sort_idxs = np.argsort(tensor.shape)
    if tensor.shape[sort_idxs[0]] > max_dim:
        return tensor, None, None

    tensor = tensor.permute(*sort_idxs.tolist())
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
    return tensor.permute(*np.argsort(sort_idxs).tolist())


class Shampoo(Transform):
    """Shampoo from Preconditioned Stochastic Tensor Optimization (https://arxiv.org/abs/1802.09568).

    .. note::
        Shampoo is usually grafted to another optimizer like Adam, otherwise it can be unstable. An example of how to do grafting is given below in the Examples section.

    .. note::
        Shampoo is a very computationally expensive optimizer, increase :code:`update_freq` if it is too slow.

    .. note::
        SOAP optimizer usually outperforms Shampoo and is also not as computationally expensive. SOAP implementation is available as :code:`tz.m.SOAP`.

    Args:
        decay (float | None, optional): slowly decays preconditioners. Defaults to None.
        beta (float | None, optional):
            if None calculates sum as in standard shampoo, otherwise uses EMA of preconditioners. Defaults to None.
        matrix_eps (float, optional): epsilon for matrix operations. Defaults to 1e-10.
        update_freq (int, optional): preconditioner update frequency. Defaults to 10.
        exp_override (int | None, optional): matrix exponent override, if not set, uses 2*ndim. Defaults to 2.
        merge_small (bool, optional): whether to merge small dims on tensors. Defaults to True.
        max_dim (int, optional): maximum dimension size for preconditioning. Defaults to 2_000.
        precondition_1d (bool, optional): whether to precondition 1d tensors. Defaults to True.
        adagrad_eps (float, optional): epsilon for adagrad division for tensors where shampoo can't be applied. Defaults to 1e-8.
        inner (Chainable | None, optional):
            module applied after updating preconditioners and before applying preconditioning.
            For example if beta≈0.999 and `inner=tz.m.EMA(0.9)`, this becomes Adam with shampoo preconditioner (ignoring debiasing).
            Defaults to None.

    Examples:
        Shampoo grafted to Adam

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.GraftModules(
                    direction = tz.m.Shampoo(),
                    magnitude = tz.m.Adam(),
                ),
                tz.m.LR(1e-3)
            )

        Adam with Shampoo preconditioner

        .. code-block:: python

            opt = tz.Modular(
                model.parameters(),
                tz.m.Shampoo(beta=0.999, inner=tz.m.EMA(0.9)),
                tz.m.Debias(0.9, 0.999),
                tz.m.LR(1e-3)
            )
    """
    def __init__(
        self,
        decay: float | None = None,
        beta: float | None = None,
        update_freq: int = 10,
        exp_override: int | None = 2,
        merge_small: bool = True,
        max_dim: int = 2_000,
        precondition_1d: bool = True,
        adagrad_eps: float = 1e-8,
        inner: Chainable | None = None,
    ):
        defaults = dict(decay=decay, beta=beta, update_freq=update_freq, exp_override=exp_override, merge_small=merge_small, max_dim=max_dim, precondition_1d=precondition_1d,adagrad_eps=adagrad_eps)
        super().__init__(defaults, uses_grad=False)

        if inner is not None:
            self.set_child('inner', inner)

    def apply_tensors(self, tensors, params, grads, loss, states, settings):
        merged_tensors = [] # target with merged dims

        # update preconditioners
        for i,(t,state, setting) in enumerate(zip(tensors, states, settings)):
            beta, update_freq, exp_override, merge_small, max_dim, precondition_1d = itemgetter(
                'beta', 'update_freq', 'exp_override', 'merge_small', 'max_dim', 'precondition_1d')(setting)

            if merge_small:
                t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, max_dim)

            merged_tensors.append(t)

            # initialize accumulators and preconditioners for each dim on 1st step
            if 'accumulators' not in state:

                if not precondition_1d and t.ndim <= 1:
                    state['accumulators'] = []

                else:
                    state['accumulators'] = [torch.eye(s, dtype=t.dtype, device=t.device) if 1<s<max_dim else None for s in t.shape]
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
                )

        # inner step
        if 'inner' in self.children:
            tensors = apply_transform(self.children['inner'], tensors, params=params, grads=grads)

            # have to merge small dims again
            merged_tensors = [] # target with merged dims
            for i,(t,state, setting) in enumerate(zip(tensors, states, settings)):
                if setting['merge_small']:
                    t, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(t, setting['max_dim'])
                merged_tensors.append(t)

        # precondition
        for i,(t,state, setting) in enumerate(zip(merged_tensors, states, settings)):
            decay, merge_small, adagrad_eps= itemgetter('decay', 'merge_small', 'adagrad_eps')(setting)

            if 'diagonal_accumulator' in state:
                tensors[i] = apply_diagonal_(t, state['diagonal_accumulator'], decay=decay, eps=adagrad_eps)
            else:
                tensors[i] = apply_shampoo_preconditioner(t, preconditioners_=state['preconditioners'], decay=decay)

            if merge_small:
                tensors[i] = _unmerge_small_dims(tensors[i], state['flat_sizes'], state['sort_idxs'])

            state['step'] += 1

        return tensors