from importlib.util import find_spec
# pylint:disable=deprecated-method
from typing import Any
from collections.abc import Sequence

import pytest
import torch

import torchzero as tz
from torchzero.utils import TensorList, vec_to_tensors

# ----------------------------------- utils ---------------------------------- #
DEVICES = ["cpu"]
if torch.cuda.is_available(): DEVICES.append("cuda")
DEVICES = tuple(DEVICES)

def _gen(device):
    return torch.Generator(device).manual_seed(0)

def cat(ts: Sequence[torch.Tensor]):
    return torch.cat([t.flatten() for t in ts])

def numel(ts: Sequence[torch.Tensor]):
    return sum(t.numel() for t in ts)

def assert_tl_equal_(tl1: Sequence[torch.Tensor | Any], tl2: Sequence[torch.Tensor | Any]):
    assert len(tl1) == len(tl2), f"TensorLists have different lengths:\n{[t.shape for t in tl1]}\n{[t.shape for t in tl2]};"
    for t1, t2 in zip(tl1, tl2):
        if t1 is None and t2 is None:
            continue
        assert t1 is not None and t2 is not None, "One tensor is None, the other is not"
        assert t1.shape == t2.shape, f"Tensors have different shapes:\n{t1}\nvs\n{t2}"
        assert torch.equal(t1, t2), f"Tensors are not equal:\n{t1}\nvs\n{t2}"

def assert_tl_allclose_(tl1: Sequence[torch.Tensor | Any], tl2: Sequence[torch.Tensor | Any], **kwargs):
    assert len(tl1) == len(tl2), f"TensorLists have different lengths:\n{[t.shape for t in tl1]}\n{[t.shape for t in tl2]};"
    for t1, t2 in zip(tl1, tl2):
        if t1 is None and t2 is None:
            continue
        assert t1 is not None and t2 is not None, "One tensor is None, the other is not"
        assert t1.shape == t2.shape, f"Tensors have different shapes:\n{t1}\nvs\n{t2}"
        assert torch.allclose(t1, t2, equal_nan=True, **kwargs), f"Tensors are not close:\n{t1}\nvs\n{t2}"

def assert_tl_same_(seq1: Sequence[torch.Tensor], seq2: Sequence[torch.Tensor]):
    seq1=tuple(seq1)
    seq2=tuple(seq2)
    assert len(seq1) == len(seq2), f'lengths do not match: {len(seq1)} != {len(seq2)}'
    for t1, t2 in zip(seq1, seq2):
        assert t1 is t2


def assert_tl_same_storage_(seq1: Sequence[torch.Tensor], seq2: Sequence[torch.Tensor]):
    seq1=tuple(seq1)
    seq2=tuple(seq2)
    assert len(seq1) == len(seq2), f'lengths do not match: {len(seq1)} != {len(seq2)}'
    for t1, t2 in zip(seq1, seq2):
        assert t1.data_ptr() == t2.data_ptr()

class _EvalCounter:
    def __init__(self, closure):
        self.closure = closure
        self.false = 0
        self.true = 0

    def __call__(self, backward=True):
        if backward: self.true += 1
        else: self.false += 1
        return self.closure(backward)

    def assert_(self, true:int, false:int):
        assert true == self.true
        assert false == self.false

    def __repr__(self):
        return f"EvalCounter(true={self.true}, false={self.false})"

# --------------------------------- objective --------------------------------

def objective_value(x:torch.Tensor, A:torch.Tensor, b:torch.Tensor):
    return 0.5 * x @ A @ x + (b @ x).exp()

def analytical_gradient(x:torch.Tensor, A:torch.Tensor, b:torch.Tensor):
    return A @ x + (b @ x).exp() * b

def analytical_hessian(x:torch.Tensor, A:torch.Tensor, b:torch.Tensor):
    return A + (b @ x).exp() * b.outer(b)

def analytical_derivative(x: torch.Tensor, b:torch.Tensor, order: int) -> torch.Tensor:
    assert order >= 3
    # n-th order outer product
    # n=4 -> 'i,j,k,l->ijkl'
    indices = 'ijklmnopqrstuvwxyz'[:order]
    b_outer = torch.einsum(f"{','.join(indices)}->{indices}", *[b] * order)
    return (b @ x).exp() * b_outer


def get_var(device, dtype=torch.float32):

    # we cat a few tensors to make sure those methods handle multiple params correctly
    p1 = torch.tensor(1., requires_grad=True, device=device, dtype=dtype)
    p2 = torch.randn(1, 3, 2, requires_grad=True, device=device, generator=_gen(device), dtype=dtype)
    p3 = torch.randn(4, requires_grad=True, device=device, generator=_gen(device), dtype=dtype)

    params = [p1, p2, p3]
    n = numel(params)

    A = torch.randn(n, n, device=device, generator=_gen(device), dtype=dtype)
    A = A.T @ A + torch.eye(n, device=device, dtype=dtype) * 1e-3
    b = torch.randn(n, device=device, generator=_gen(device), dtype=dtype)

    def closure(backward=True):
        x = cat(params)
        loss = objective_value(x, A, b)

        if backward:
            for p in params:
                p.grad = None
            loss.backward()

        return loss

    objective = _EvalCounter(closure)
    var = tz.core.Objective(params=params, closure=objective, model=None, current_step=0)

    return var, A, b, objective

# ------------------------------------ hvp ----------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
def test_gradient(device):
    """makes sure gradient is correct"""
    var, A, b, objective = get_var(device)
    grad = var.get_grads()
    assert torch.allclose(cat(grad), analytical_gradient(cat(var.params), A, b))
    objective.assert_(true=1, false=0)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd"])
@pytest.mark.parametrize("get_grad", [True, False])
def test_hvp_autograd(device, at_x0, hvp_method, get_grad):
    """compares hessian-vector product with analytical"""

    var, A, b, objective = get_var(device)

    grad = None
    if get_grad:
        grad = var.get_grads(create_graph=True, at_x0=at_x0) # one false (one closure call with backward=False)

    # generate random z
    n = numel(var.params)
    z = vec_to_tensors(torch.randn(n, device=device, generator=_gen(device)), var.params)

    # Hz
    # this is for all following autograd tests
    # if at_x0:
    #   one false call happens either in get_grad or here, so 1 false
    # else:
    #   if get_grad, both get_grad and this call with false, so 2 false
    #   else only this calls with false, so 1 false
    Hz, rgrad = var.hessian_vector_product(z, None, at_x0=at_x0, hvp_method=hvp_method, h=1e-3)

    # check storage
    assert rgrad is not None
    if at_x0:
        assert var.grads is not None
        assert_tl_same_(var.grads, rgrad)
        if grad is not None: assert_tl_same_(grad, rgrad)
    else:
        assert var.grads is None
        if grad is not None: assert_tl_allclose_(grad, rgrad)

    # check against known Hvp
    x = cat(var.params)
    assert torch.allclose(cat(rgrad), analytical_gradient(x, A, b))
    assert torch.allclose(cat(Hz), analytical_hessian(x, A, b) @ cat(z))

    # check evals
    if at_x0: false = 1
    else:
        if get_grad: false = 2
        else: false = 1
    objective.assert_(true=0, false=false)

# -------------------------- hessian-matrix product -------------------------- #\
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd"])
@pytest.mark.parametrize("get_grad", [True, False])
def test_hessian_matrix_product(device, at_x0, hvp_method, get_grad):
    """compares hessian-matrix product with analytical"""

    var, A, b, objective = get_var(device)
    if get_grad:
        var.get_grads(create_graph=True, at_x0=at_x0) # one false

    # generate random matrix
    n = numel(var.params)
    Z = torch.randn(n, n*2, device=device, generator=_gen(device))

    # HZ same as above
    HZ, rgrad = var.hessian_matrix_product(Z, rgrad=None, at_x0=at_x0, hvp_method=hvp_method, h=1e-3)

    # check storage
    assert rgrad is not None
    if at_x0:
        assert var.grads is not None
        assert_tl_same_(rgrad, var.grads)
    else:
        assert var.grads is None

    # check against known HZ
    x = cat(var.params)
    assert torch.allclose(HZ, analytical_hessian(x, A, b) @ Z, rtol=1e-4, atol=1e-6), f"{HZ = }, {A@Z = }"

    # check evals
    if at_x0: false = 1
    else:
        if get_grad: false = 2
        else: false = 1
    objective.assert_(true=0, false=false)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd", "fd_forward", "fd_central"])
@pytest.mark.parametrize("h", [1e-1, 1e-2, 1e-3])
def test_hessian_vector_vs_matrix_product(device, at_x0, hvp_method, h):
    """compares hessian_vector_product and hessian_matrix_product, including fd"""

    var, A, b, objective = get_var(device, dtype=torch.float64)

    # generate random matrix
    n = numel(var.params)
    Z = torch.randn(n, n*2, device=device, generator=_gen(device))
    z_vecs = [vec_to_tensors(col, var.params) for col in Z.unbind(1)]

    # hessian-vector
    rgrad = None
    Hzs = []
    for z in z_vecs:
        Hz, rgrad = var.hessian_vector_product(z, rgrad=rgrad, at_x0=at_x0, hvp_method=hvp_method, h=h, retain_graph=True)
        Hzs.append(cat(Hz))

    # check evals (did n*2 hvps)
    if hvp_method in ('autograd', 'batched_autograd'): objective.assert_(true=0, false=1)
    elif hvp_method == 'fd_central': objective.assert_(true=n*4, false=0)
    elif hvp_method == 'fd_forward': objective.assert_(true=n*2+1, false=0)
    else: assert False, hvp_method

    # clear evals
    objective.true = objective.false = 0

    # hessian-matrix
    HZ, rgrad = var.hessian_matrix_product(Z, rgrad=rgrad, at_x0=at_x0, hvp_method=hvp_method, h=h)

    # check evals (did n*2 hvps, initial grad is rgrad)
    if hvp_method in ('autograd', 'batched_autograd'): objective.assert_(true=0, false=0)
    elif hvp_method == 'fd_central': objective.assert_(true=n*4, false=0)
    elif hvp_method == 'fd_forward': objective.assert_(true=n*2, false=0)
    else: assert False, hvp_method

    # check storage
    if hvp_method == 'fd_central': assert rgrad is None
    else: assert rgrad is not None

    if at_x0:
        if hvp_method == 'fd_central':  assert var.grads is None
        else:
            assert var.grads is not None
            assert rgrad is not None
            assert_tl_same_(rgrad, var.grads)
    else:
        assert var.grads is None

    # check that they match
    assert torch.allclose(HZ, torch.stack(Hzs, dim=-1)), f"{HZ = }, {torch.stack(Hzs, dim=-1) = }"

# -------------------------------- hutchinson -------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd"])
@pytest.mark.parametrize("zHz", [True, False])
@pytest.mark.parametrize("get_grad", [True, False])
def test_hutchinson(device, at_x0, hvp_method, zHz, get_grad):
    """compares autograd hutchinson with one computed with analytical hessian-vector products"""

    var, A, b, objective = get_var(device)
    if get_grad:
        var.get_grads(create_graph=True, at_x0=at_x0) # one false

    # 10 random vecs
    n = numel(var.params)
    zs = [vec_to_tensors(torch.randn(n, device=device, generator=_gen(device)), var.params) for _ in range(10)]

    # compute hutchinson estimate, same as above
    D, rgrad = var.hutchinson_hessian(rgrad=None, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method=hvp_method, h=1e-3, zHz=zHz, generator=None)

    # check storage
    assert rgrad is not None
    if at_x0:
        assert var.grads is not None
        if at_x0: assert_tl_same_(var.grads, rgrad)
    else:
        assert var.grads is None

    # compute D via known hvp
    x = cat(var.params)
    z_vecs = [cat(z) for z in zs]
    Hzs = [analytical_hessian(x, A, b) @ z for z in z_vecs]
    D2 = torch.stack(Hzs)
    if zHz: D2 *= torch.stack(z_vecs)
    D2 = D2.mean(0)

    # compare Ds
    assert_tl_allclose_(D, vec_to_tensors(D2, var.params))

    # check evals
    if at_x0: false = 1
    else:
        if get_grad: false = 2
        else: false = 1
    objective.assert_(true=0, false=false)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("zHz", [True, False])
@pytest.mark.parametrize("get_grad", [True, False])
@pytest.mark.parametrize("pass_rgrad", [True, False])
def test_hutchinson_batching(device, at_x0, zHz, get_grad, pass_rgrad):
    """compares batched and unbatched hutchinson"""

    var, A, b, objective = get_var(device)
    if get_grad:
        var.get_grads(create_graph=True, at_x0=at_x0) # one false

    # 10 random vecs
    n = numel(var.params)
    zs = [vec_to_tensors(torch.randn(n, device=device, generator=_gen(device)), var.params) for _ in range(10)]

    # compute hutchinson estimate, same as above
    D, rgrad = var.hutchinson_hessian(rgrad=None, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method='autograd', h=1e-3, zHz=zHz, generator=None, retain_graph=True)

    # check evals
    if at_x0: false = 1
    else:
        if get_grad: false = 2
        else: false = 1
    objective.assert_(true=0, false=false)

    # reset evals
    objective.true = objective.false = 0

    # compute batched hutchinson estimate, if not at x0, one false if not pass_rgrad
    D2, rgrad2 = var.hutchinson_hessian(rgrad=rgrad if pass_rgrad else None, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method='batched_autograd', h=1e-3, zHz=zHz, generator=None)

    # check storage
    assert rgrad is not None
    assert rgrad2 is not None
    if at_x0:
        assert var.grads is not None
        assert_tl_same_(var.grads, rgrad2)
    else:
        assert var.grads is None
    if at_x0 or pass_rgrad: assert_tl_same_(rgrad, rgrad2)

    # make sure Ds match
    assert_tl_allclose_(D, D2)

    # check evals
    if at_x0 or pass_rgrad: false = 0
    else: false = 1
    objective.assert_(true=0, false=false)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd"])
@pytest.mark.parametrize("hvp_fd_method", ["fd_forward", "fd_central"])
@pytest.mark.parametrize("zHz", [True, False])
def test_hutchinson_fd(device, at_x0, hvp_method, hvp_fd_method, zHz):
    """compares exact and FD hutchinson"""

    var, A, b, objective = get_var(device)

    # 10 random vecs
    n = numel(var.params)
    zs = [vec_to_tensors(torch.randn(n, device=device, generator=_gen(device)), var.params) for _ in range(10)]

    # compute hutchinson D, always one false
    D, rgrad = var.hutchinson_hessian(rgrad=None, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method=hvp_method, h=1e-3, zHz=zHz, generator=None)

    # compute finite difference hutchinson D
    # rgrad is already computed
    # fd_forward 10 true, fd_central 20 true
    D_fd, rgrad = var.hutchinson_hessian(rgrad=rgrad, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method=hvp_fd_method, h=1e-3, zHz=zHz, generator=None)

    # make sure they are close
    assert_tl_allclose_(D, D_fd, rtol=1e-2, atol=1e-2)

    # check evals
    assert objective.false == 1
    if hvp_fd_method == 'fd_forward':
        assert objective.true == 10
    else:
        assert objective.true == 20



@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hvp_method", ["autograd", "batched_autograd", "fd_forward", "fd_central"])
@pytest.mark.parametrize("h", [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize("zHz", [True, False])
@pytest.mark.parametrize("get_grad", [True, False])
@pytest.mark.parametrize("pass_rgrad", [True, False])
def test_hvp_vs_hutchinson(device, at_x0, hvp_method, h, zHz, get_grad, pass_rgrad):
    """compares hutchinson via hessian_vector_product and via hutchinson methods, including fd"""

    var, A, b, objective = get_var(device)
    if get_grad:
        var.get_grads(create_graph=hvp_method in ("autograd", "batched_autograd"), at_x0=at_x0) # one false or true

    # generate 10 vecs
    n = numel(var.params)
    zs = [vec_to_tensors(torch.randn(n, device=device, generator=_gen(device)), var.params) for _ in range(10)]

    # mean of 10 z * Hz
    # autograd and batched autograd - same as above
    # fd forward
    #   if at_x0, first true either here or in get_grad, then 10 true, so total always 11 true
    #   else extra true in get_grad so 12 true
    # fd central - 20 true plus one if get_grad
    D = [torch.zeros_like(t) for t in var.params]
    rgrad = None
    for z in zs:
        Hz, rgrad = var.hessian_vector_product(z, rgrad, at_x0=at_x0, hvp_method=hvp_method, h=h, retain_graph=True)

        if zHz: torch._foreach_mul_(Hz, z)
        torch._foreach_add_(D, Hz, alpha = 1/10)

    # check storage
    if not at_x0: assert var.grads is None
    else:
        if hvp_method == 'fd_central':
            assert rgrad is None
            if get_grad: assert var.grads is not None

        else:
            assert var.grads is not None
            assert rgrad is not None
            assert_tl_same_(var.grads, rgrad)

    # check number of evals
    if hvp_method in ('autograd',  'batched_autograd'):
        if at_x0: false = 1
        else:
            if get_grad: false = 2
            else: false = 1
        objective.assert_(true=0, false=false)

    elif hvp_method == "fd_forward":
        if get_grad and not at_x0: true = 12
        else: true = 11
        objective.assert_(true=true, false=0)

    elif hvp_method == 'fd_central':
        if get_grad: objective.assert_(true=21, false=0)
        else: objective.assert_(true=20, false=0)

    else:
        assert False, hvp_method

    # reset evals
    objective.true = objective.false = 0

    # compute hutchinson hessian
    # number of evals
    # autograd/batched autograd - one false only if both pass_rgrad and at_x0 are False, else 0
    # fd_forward - 11 true if both pass_rgrad and at_x0 are False, else 10 true
    # fd_central - always 20 true
    D2, rgrad2 = var.hutchinson_hessian(rgrad=rgrad if pass_rgrad else None, at_x0=at_x0, n_samples=None, distribution=zs, hvp_method=hvp_method, h=h, zHz=zHz, generator=None)

    # check storage
    if hvp_method != "fd_central":
        assert rgrad is not None
        assert rgrad2 is not None
        if at_x0 or pass_rgrad: assert_tl_same_(rgrad, rgrad2)
        else: assert_tl_allclose_(rgrad, rgrad2)

    # check that Ds match
    assert_tl_allclose_(D, D2)

    # check evals
    # check number of evals
    if hvp_method in ('autograd',  'batched_autograd'):
        if at_x0 or pass_rgrad: false = 0
        else: false = 1
        objective.assert_(true=0, false=false)

    elif hvp_method == "fd_forward":
        if at_x0 or pass_rgrad: objective.assert_(true=10, false=0)
        else: objective.assert_(true=11, false=0)
    elif hvp_method == 'fd_central':
        objective.assert_(true=20, false=0)
    else:
        assert False, hvp_method

    # update should be none after all of this
    assert var.updates is None

_HESSIAN_METHODS = [
    "batched_autograd",
    "autograd",
    "functional_revrev",
    # "functional_fwdrev", # has shape issue
    "func",
    "gfd_forward",
    "gfd_central",
    "fd",
    "fd_full",
]

# if find_spec("thoad") is not None: _HESSIAN_METHODS.append("thoad")
# SqueezeBackward4 is not supported.

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("at_x0", [True, False])
@pytest.mark.parametrize("hessian_method", _HESSIAN_METHODS)
def test_hessian(device, at_x0, hessian_method):
    """compares hessian with analytical, including gfd and fd"""

    var, A, b, objective = get_var(device, dtype=torch.float64)
    n = numel(var.params)

    # compute hessian
    if hessian_method in ("fd", "fd_full"): h = 1e-2
    else: h = 1e-5
    f, g_list, H = var.hessian(hessian_method=hessian_method, h=h, at_x0=at_x0)

    # check storages
    if hessian_method in ("batched_autograd", "autograd", "gfd_forward", "fd", "fd_full"):
        if hessian_method == "gfd_forward": assert f is None
        else: assert f == objective.closure(False)
        assert g_list is not None
        if at_x0:
            assert var.grads is not None
            assert_tl_same_(g_list, var.grads)
        else:
            assert var.grads is None
    else:
        assert f is None
        assert g_list is None
        assert var.grads is None

    # compare with analytical
    x = cat(var.params)
    H_real = analytical_hessian(x, A, b)
    if hessian_method in ("gfd_forward", "gfd_central"):
        assert torch.allclose(H, H_real, rtol=1e-1, atol=1e-1), f"{H = }, {H_real = }"

    elif hessian_method in ("fd", "fd_full"):
        # assert torch.allclose(H, H_real, rtol=1e-1, atol=1e-1), f"{H = }, {H_real = }"
        # TODO find a good test

        # compare gradient with analytical
        g_real = analytical_gradient(x, A, b)
        assert g_list is not None
        assert torch.allclose(cat(g_list), g_real, rtol=1e-2, atol=1e-2), f"{cat(g_list) = }, {g_real = }"

    else:
        assert torch.allclose(H, H_real), f"{H = }, {H_real = }"


    # check evals
    if hessian_method == "gfd_forward":
        objective.assert_(true=n+1, false=0)

    elif hessian_method == "gfd_central":
        objective.assert_(true=n*2, false=0)

    elif hessian_method == "fd":
        objective.assert_(true=0, false=2*n**2 + 1)

    elif hessian_method == "fd_full":
        objective.assert_(true=0, false=4*n**2 - 2*n + 1)

    else:
        objective.assert_(true=0, false=1)
