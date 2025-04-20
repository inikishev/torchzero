from . import tensorlist as tl
from .compile import _optional_compiler, benchmark_compile_cpu, benchmark_compile_cuda
from .numberlist import NumberList
from .optimizer import (
    Init,
    ListLike,
    Optimizer,
    ParamFilter,
    get_group_vals,
    get_params,
    get_state_vals,
    grad_at_params,
    grad_vec_at_params,
    loss_at_params,
    loss_grad_at_params,
    loss_grad_vec_at_params,
)
from .params import (
    Params,
    _add_defaults_to_param_groups_,
    _add_updates_grads_to_param_groups_,
    _copy_param_groups,
    _make_param_groups,
)
from .python_tools import FallbackDict, flatten, generic_eq, reduce_dim
from .tensorlist import TensorList, as_tensorlist, Distributions
from .torch_tools import tofloat, tolist, tonumpy, totensor, vec_to_tensors, vec_to_tensors_


def set_compilation(enable: bool):
    """`enable` is False by default. When True, certain functions will be compiled, which may not work on some systems like Windows, but it usually improves performance."""
    _optional_compiler.enable = enable

def _maybe_compile(fn): return _optional_compiler.compile(fn)