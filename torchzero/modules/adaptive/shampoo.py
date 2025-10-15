from collections.abc import Sequence, Iterable

import numpy as np
import torch

from ...core import Chainable, TensorTransform
from ...linalg.matrix_power import MatrixPowerMethod, matrix_power as _matrix_power
from ...utils import set_storage_


def update_shampoo_preconditioner_(
    grad: torch.Tensor,
    accumulators_: list[torch.Tensor | None],
    preconditioners_: list[torch.Tensor | None],
    step: int,
    precond_freq: int,
    matrix_power: float | None,
    beta: float | None,
    reg: float,
    matrix_power_method: MatrixPowerMethod,
):
    for i, (accumulator, preconditioner) in enumerate(zip(accumulators_, preconditioners_)):
        if accumulator is None: continue
        assert preconditioner is not None

        axes = list(range(i)) + list(range(i + 1, grad.ndim))
        if beta is None: accumulator.add_(torch.tensordot(grad, grad, (axes, axes))) # pyright:ignore[reportArgumentType]
        else: accumulator.lerp_(torch.tensordot(grad, grad, (axes, axes)), 1-beta) # pyright:ignore[reportArgumentType]

        if step % precond_freq == 0:
            if reg != 0:
                accumulator = accumulator + torch.eye(accumulator.size(0), device=accumulator.device, dtype=accumulator.dtype).mul_(reg)

            if matrix_power is None: matrix_power = -1 / max(grad.ndim, 2)
            set_storage_(preconditioner, _matrix_power(accumulator, matrix_power, method=matrix_power_method))

def apply_shampoo_preconditioner(
    tensor: torch.Tensor,
    preconditioners_: list[torch.Tensor | None],
):
    for i, preconditioner in enumerate(preconditioners_):
        if preconditioner is None: continue
        tensor = torch.tensordot(tensor, preconditioner, ([0], [0])) # pyright:ignore[reportArgumentType]
    return tensor


def update_diagonal_(grad: torch.Tensor, diagonal_accumulator_: torch.Tensor, beta: float | None):
    if beta is None: diagonal_accumulator_.add_(grad.pow(2))
    else: diagonal_accumulator_.mul_(beta).addcmul_(grad, grad, value=1-beta)

def apply_diagonal_(grad_: torch.Tensor, diagonal_accumulator_: torch.Tensor, eps: float):
    grad_.div_(diagonal_accumulator_.sqrt() + eps)
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

def diagonal_memory(params: torch.nn.Module | torch.Tensor | Iterable[torch.Tensor]):
    """computes number of parameters"""
    if isinstance(params, torch.nn.Module): params = params.parameters()
    if isinstance(params, torch.Tensor): params = [params,]
    params = list(params)
    return sum(p.numel() for p in params)

def kronecker_memory(params: torch.nn.Module | torch.Tensor | Iterable[torch.Tensor], merge_small:bool=True, max_dim:int=10_000):
    """computes total size of tensors required to store shampoo preconditioner"""
    if isinstance(params, torch.nn.Module): params = params.parameters()
    if isinstance(params, torch.Tensor): params = [params,]
    params = list(params)

    memory = 0
    for p in params:
        if merge_small:
            p, _, _ = _merge_small_dims(p, max_dim)
        for dim in p.size():
            if dim > max_dim: memory += dim
            else: memory += dim**2

    return memory




class Shampoo(TensorTransform):
    """Shampoo from Preconditioned Stochastic Tensor Optimization (https://arxiv.org/abs/1802.09568).

    Notes:
        Shampoo is usually grafted to another optimizer like Adam, otherwise it can be unstable. An example of how to do grafting is given below in the Examples section.

        Shampoo is a very computationally expensive optimizer, increase ``update_freq`` if it is too slow.

        SOAP optimizer usually outperforms Shampoo and is also not as computationally expensive. SOAP implementation is available as ``tz.m.SOAP``.

    Args:
        update_freq (int, optional): preconditioner update frequency. Defaults to 10.
        matrix_power (float | None, optional): overrides matrix exponent. By default uses ``-1/grad.ndim``. Defaults to None.
        merge_small (bool, optional): whether to merge small dims on tensors. Defaults to True.
        max_dim (int, optional): maximum dimension size for preconditioning. Defaults to 10_000.
        precondition_1d (bool, optional): whether to precondition 1d tensors. Defaults to True.
        adagrad_eps (float, optional): epsilon for adagrad division for tensors where shampoo can't be applied. Defaults to 1e-8.
        matrix_power_method (MatrixPowerMethod, optional): how to compute matrix power.
        beta (float | None, optional):
            if None calculates sum as in standard Shampoo, otherwise uses EMA of preconditioners. Defaults to None.
        inner (Chainable | None, optional):
            module applied after updating preconditioners and before applying preconditioning.
            For example if beta≈0.999 and `inner=tz.m.EMA(0.9)`, this becomes Adam with shampoo preconditioner (ignoring debiasing).
            Defaults to None.

    Examples:
    Shampoo grafted to Adam

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.GraftModules(
            direction = tz.m.Shampoo(),
            magnitude = tz.m.Adam(),
        ),
        tz.m.LR(1e-3)
    )
    ```

    Adam with Shampoo preconditioner

    ```python
    opt = tz.Optimizer(
        model.parameters(),
        tz.m.Shampoo(beta=0.999, inner=tz.m.EMA(0.9)),
        tz.m.Debias(0.9, 0.999),
        tz.m.LR(1e-3)
    )
    ```
    """
    def __init__(
        self,
        reg: float = 1e-12,
        precond_freq: int = 10,
        matrix_power: float | None = None,
        merge_small: bool = True,
        max_dim: int = 10_000,
        precondition_1d: bool = True,
        adagrad_eps: float = 1e-8,
        matrix_power_method: MatrixPowerMethod = "eigh_abs",
        beta: float | None = None,
        beta_debias: bool = True,

        inner: Chainable | None = None,
    ):
        defaults = locals().copy()
        del defaults['self'], defaults["inner"]

        super().__init__(defaults, inner=inner)

    @torch.no_grad
    def single_tensor_initialize(self, tensor, param, grad, loss, state, setting):
        if setting["merge_small"]:
            tensor, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(tensor, setting["max_dim"])

        if tensor.ndim <= 1 and not setting["precondition_1d"]:
            state["accumulators"] = []

        else:
            max_dim = setting["max_dim"]
            state['accumulators'] = [
                torch.eye(s, dtype=tensor.dtype, device=tensor.device) if 1<s<max_dim else None for s in tensor.shape
            ]
            state['preconditioners'] = [
                torch.eye(s, dtype=tensor.dtype, device=tensor.device) if 1<s<max_dim else None for s in tensor.shape
            ]

        # either scalar parameter, 1d with precondition_1d=False, or too big, then diagonal preconditioner is used.
        if len([i is not None for i in state['accumulators']]) == 0:
            state['diagonal_accumulator'] = torch.zeros_like(tensor)

        state['step'] = 0
        state["num_GTG"] = 0

    @torch.no_grad
    def single_tensor_update(self, tensor, param, grad, loss, state, setting):
        if setting["merge_small"]:
            tensor, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(tensor, setting["max_dim"])

            if "inner" not in self.children:
                state["merged"] = tensor

        if 'diagonal_accumulator' in state:
            update_diagonal_(tensor, state['diagonal_accumulator'], beta=setting["beta"])
        else:
            update_shampoo_preconditioner_(
                tensor,
                accumulators_=state['accumulators'],
                preconditioners_=state['preconditioners'],
                step=state['step'],
                precond_freq=setting["precond_freq"],
                matrix_power=setting["matrix_power"],
                beta=setting["beta"],
                reg=setting["reg"],
                matrix_power_method=setting["matrix_power_method"],
            )

        if state["step"] % setting["precond_freq"] == 0:
            state["num_GTG"] += 1

        state["step"] += 1


    @torch.no_grad
    def single_tensor_apply(self, tensor, param, grad, loss, state, setting):

        if setting["merge_small"]:
            if "inner" not in self.children:
                tensor = state.pop("merged")
            else:
                tensor, state['flat_sizes'], state['sort_idxs'] = _merge_small_dims(tensor, setting["max_dim"])

        if 'diagonal_accumulator' in state:
            dir = apply_diagonal_(tensor, state['diagonal_accumulator'], eps=setting["adagrad_eps"])
        else:
            dir = apply_shampoo_preconditioner(tensor, preconditioners_=state['preconditioners'])

        if setting["merge_small"]:
            dir = _unmerge_small_dims(dir, state['flat_sizes'], state['sort_idxs'])

        if setting['beta_debias'] and setting["beta"] is not None:
            bias_correction = 1 - (setting["beta"] ** state["num_GTG"])
            dir *= bias_correction ** 0.5

        return dir

