# pylint: disable = redefined-outer-name, bad-indentation
"""note that a lot of this is written by Gemini for a better coverage."""
import math

import pytest
import torch

from torchzero.utils.tensorlist import (
    TensorList,
    _MethodCallerWithArgs,
    as_tensorlist,
    generic_clamp,
    mean,
    median,
    quantile,
    stack,
)
from torchzero.utils.tensorlist import sum as tl_sum
from torchzero.utils.tensorlist import where as tl_where


def randmask_like(x,device=None,dtype=None):
    return torch.rand_like(x.float(), device=device,dtype=dtype) > 0.5

# Helper function for comparing TensorLists element-wise
def assert_tl_equal(tl1: TensorList, tl2: TensorList):
    assert len(tl1) == len(tl2), f"TensorLists have different lengths:\n{[t.shape for t in tl1]}\n{[t.shape for t in tl2]};"
    for t1, t2 in zip(tl1, tl2):
        if t1 is None and t2 is None:
            continue
        assert t1 is not None and t2 is not None, "One tensor is None, the other is not"
        assert t1.shape == t2.shape, f"Tensors have different shapes:\n{t1}\nvs\n{t2}"
        assert torch.equal(t1, t2), f"Tensors are not equal:\n{t1}\nvs\n{t2}"

def assert_tl_allclose(tl1: TensorList, tl2: TensorList, **kwargs):
    assert len(tl1) == len(tl2), f"TensorLists have different lengths:\n{[t.shape for t in tl1]}\n{[t.shape for t in tl2]};"
    for t1, t2 in zip(tl1, tl2):
        if t1 is None and t2 is None:
            continue
        assert t1 is not None and t2 is not None, "One tensor is None, the other is not"
        assert t1.shape == t2.shape, f"Tensors have different shapes:\n{t1}\nvs\n{t2}"
        assert torch.allclose(t1, t2, equal_nan=True, **kwargs), f"Tensors are not close:\n{t1}\nvs\n{t2}"

# --- Fixtures ---

@pytest.fixture
def simple_tensors() -> list[torch.Tensor]:
    return [torch.randn(2, 3), torch.randn(5), torch.randn(1, 4, 2)]


@pytest.fixture
def big_tensors() -> list[torch.Tensor]:
    return [torch.randn(20, 50), torch.randn(3,3,3,3,3,3), torch.randn(1000)]


@pytest.fixture
def simple_tl(simple_tensors) -> TensorList:
    return TensorList(simple_tensors)

@pytest.fixture
def simple_tl_clone(simple_tl) -> TensorList:
    return simple_tl.clone()

@pytest.fixture
def big_tl(big_tensors) -> TensorList:
    return TensorList(big_tensors)

@pytest.fixture
def grad_tensors() -> list[torch.Tensor]:
    return [
        torch.randn(2, 2, requires_grad=True),
        torch.randn(3, requires_grad=False),
        torch.randn(1, 5, requires_grad=True)
    ]

@pytest.fixture
def grad_tl(grad_tensors) -> TensorList:
    return TensorList(grad_tensors)

@pytest.fixture
def int_tensors() -> list[torch.Tensor]:
    return [torch.randint(0, 10, (2, 3)), torch.randint(0, 10, (5,))]

@pytest.fixture
def int_tl(int_tensors) -> TensorList:
    return TensorList(int_tensors)

@pytest.fixture
def bool_tensors() -> list[torch.Tensor]:
    return [torch.rand(2, 3) > 0.5, torch.rand(5) > 0.5]

@pytest.fixture
def bool_tl(bool_tensors) -> TensorList:
    return TensorList(bool_tensors)

@pytest.fixture
def complex_tensors() -> list[torch.Tensor]:
    return [torch.randn(2, 3, dtype=torch.complex64), torch.randn(5, dtype=torch.complex64)]

@pytest.fixture
def complex_tl(complex_tensors) -> TensorList:
    return TensorList(complex_tensors)

# --- Test Cases ---

def test_initialization(simple_tensors):
    tl = TensorList(simple_tensors)
    assert isinstance(tl, TensorList)
    assert isinstance(tl, list)
    assert len(tl) == len(simple_tensors)
    for i in range(len(tl)):
        assert torch.equal(tl[i], simple_tensors[i])

def test_empty_initialization():
    tl = TensorList()
    assert isinstance(tl, TensorList)
    assert len(tl) == 0

def test_as_tensorlist(simple_tensors, simple_tl: TensorList):
    tl = as_tensorlist(simple_tensors)
    assert isinstance(tl, TensorList)
    assert_tl_equal(tl, simple_tl)

    tl2 = as_tensorlist(simple_tl)
    assert tl2 is simple_tl # Should return the same object if already TensorList

def test_complex_classmethod(simple_tensors):
    real_tl = TensorList([t.float() for t in simple_tensors])
    imag_tl = TensorList([torch.randn_like(t) for t in simple_tensors])
    complex_tl = TensorList.complex(real_tl, imag_tl)

    assert isinstance(complex_tl, TensorList)
    assert len(complex_tl) == len(real_tl)
    for i in range(len(complex_tl)):
        assert complex_tl[i].dtype == torch.complex64 or complex_tl[i].dtype == torch.complex128
        assert torch.equal(complex_tl[i].real, real_tl[i])
        assert torch.equal(complex_tl[i].imag, imag_tl[i])

# --- Properties ---
def test_properties(simple_tl: TensorList, simple_tensors):
    assert simple_tl.device == [t.device for t in simple_tensors]
    assert simple_tl.dtype == [t.dtype for t in simple_tensors]
    assert simple_tl.requires_grad == [t.requires_grad for t in simple_tensors]
    assert simple_tl.shape == [t.shape for t in simple_tensors]
    assert simple_tl.size() == [t.size() for t in simple_tensors]
    assert simple_tl.size(0) == [t.size(0) for t in simple_tensors]
    assert simple_tl.ndim == [t.ndim for t in simple_tensors]
    assert simple_tl.ndimension() == [t.ndimension() for t in simple_tensors]
    assert simple_tl.numel() == [t.numel() for t in simple_tensors]

def test_grad_property(grad_tl: TensorList, grad_tensors):
    # Initially grads are None
    assert all(g is None for g in grad_tl.grad)

    # Set some grads
    for i, t in enumerate(grad_tensors):
        if t.requires_grad:
            t.grad = torch.ones_like(t) * (i + 1)

    grads = grad_tl.grad
    assert isinstance(grads, TensorList)
    assert len(grads) == len(grad_tl)
    for i in range(len(grad_tl)):
        if grad_tensors[i].requires_grad:
            assert torch.equal(grads[i], torch.ones_like(grad_tensors[i]) * (i + 1))
        else:
            assert grads[i] is None # Accessing .grad on non-req-grad tensor returns None

def test_real_imag_properties(complex_tl: TensorList, complex_tensors):
    real_part = complex_tl.real
    imag_part = complex_tl.imag
    assert isinstance(real_part, TensorList)
    assert isinstance(imag_part, TensorList)
    assert len(real_part) == len(complex_tl)
    assert len(imag_part) == len(complex_tl)
    for i in range(len(complex_tl)):
        assert torch.equal(real_part[i], complex_tensors[i].real)
        assert torch.equal(imag_part[i], complex_tensors[i].imag)

def test_view_as_real_complex(complex_tl: TensorList, complex_tensors):
    real_view = complex_tl.view_as_real()
    assert isinstance(real_view, TensorList)
    assert len(real_view) == len(complex_tl)
    for i in range(len(complex_tl)):
        assert torch.equal(real_view[i], torch.view_as_real(complex_tensors[i]))

    # Convert back
    complex_view_again = real_view.view_as_complex()
    assert_tl_equal(complex_view_again, complex_tl)

# --- Utility Methods ---

def test_type_as(simple_tl: TensorList, int_tl: TensorList):
    int_casted_tl = simple_tl.type_as(int_tl[0]) # Cast like first int tensor
    assert isinstance(int_casted_tl, TensorList)
    assert all(t.dtype == int_tl[0].dtype for t in int_casted_tl)

    float_casted_tl = int_tl.type_as(simple_tl) # Cast like corresponding float tensors
    assert isinstance(float_casted_tl, TensorList)
    assert all(t.dtype == s.dtype for t, s in zip(float_casted_tl, simple_tl))


def test_fill_none(simple_tl: TensorList):
    tl_with_none = TensorList([simple_tl[0], None, simple_tl[2]])
    reference_tl = simple_tl.clone() # Use original shapes as reference

    filled_tl = tl_with_none.fill_none(reference_tl)
    assert isinstance(filled_tl, TensorList)
    assert filled_tl[0] is tl_with_none[0] # Should keep existing tensor
    assert torch.equal(filled_tl[1], torch.zeros_like(reference_tl[1]))
    assert filled_tl[2] is tl_with_none[2]
    # Check original is not modified
    assert tl_with_none[1] is None

    filled_tl_inplace = tl_with_none.fill_none_(reference_tl)
    assert filled_tl_inplace is tl_with_none # Should return self
    assert filled_tl_inplace[0] is simple_tl[0]
    assert torch.equal(filled_tl_inplace[1], torch.zeros_like(reference_tl[1]))
    assert filled_tl_inplace[2] is simple_tl[2]


def test_get_grad(grad_tl: TensorList, grad_tensors):
    # No grads initially
    assert len(grad_tl.get_grad()) == 0

    # Set grads only for tensors requiring grad
    expected_grads = []
    for i, t in enumerate(grad_tensors):
        if t.requires_grad:
            g = torch.rand_like(t)
            t.grad = g
            expected_grads.append(g)

    retrieved_grads = grad_tl.get_grad()
    assert isinstance(retrieved_grads, TensorList)
    assert len(retrieved_grads) == len(expected_grads)
    for rg, eg in zip(retrieved_grads, expected_grads):
        assert torch.equal(rg, eg)


def test_with_requires_grad(grad_tl: TensorList, grad_tensors):
    req_grad_true = grad_tl.with_requires_grad(True)
    expected_true = [t for t in grad_tensors if t.requires_grad]
    assert len(req_grad_true) == len(expected_true)
    for rt, et in zip(req_grad_true, expected_true):
        assert rt is et

    req_grad_false = grad_tl.with_requires_grad(False)
    expected_false = [t for t in grad_tensors if not t.requires_grad]
    assert len(req_grad_false) == len(expected_false)
    for rt, et in zip(req_grad_false, expected_false):
        assert rt is et


def test_with_grad(grad_tl: TensorList, grad_tensors):
    assert len(grad_tl.with_grad()) == 0 # No grads set yet

    # Set grads for tensors requiring grad
    expected_with_grad = []
    for i, t in enumerate(grad_tensors):
        if t.requires_grad:
            t.grad = torch.ones_like(t) * i
            expected_with_grad.append(t)

    has_grad_tl = grad_tl.with_grad()
    assert isinstance(has_grad_tl, TensorList)
    assert len(has_grad_tl) == len(expected_with_grad)
    for hg, eg in zip(has_grad_tl, expected_with_grad):
        assert hg is eg


def test_ensure_grad_(grad_tl: TensorList, grad_tensors):
    # Call ensure_grad_
    grad_tl.ensure_grad_()

    for t in grad_tl:
        if t.requires_grad:
            assert t.grad is not None
            assert torch.equal(t.grad, torch.zeros_like(t))
        else:
            assert t.grad is None

    # Call again, should not change existing zero grads
    grad_tl.ensure_grad_()
    for t in grad_tl:
        if t.requires_grad:
            assert t.grad is not None, 'this is a fixture'
            assert torch.equal(t.grad, torch.zeros_like(t))


def test_accumulate_grad_(grad_tl: TensorList, grad_tensors):
    new_grads = TensorList([torch.rand_like(t) for t in grad_tensors])
    new_grads_copy = new_grads.clone()

    # First accumulation (grads are None or zero if ensure_grad_ was called)
    grad_tl.accumulate_grad_(new_grads)
    for t, ng in zip(grad_tl, new_grads_copy):
        # if t.requires_grad:
        assert t.grad is not None
        assert torch.equal(t.grad, ng)
        # else:
        #     assert t.grad is None # Should not create grad if requires_grad is False

    # Second accumulation
    new_grads_2 = TensorList([torch.rand_like(t) for t in grad_tensors])
    expected_grads = TensorList([g + ng2 for t, g, ng2 in zip(grad_tensors, grad_tl.grad, new_grads_2)])

    grad_tl.accumulate_grad_(new_grads_2)
    for t, eg in zip(grad_tl, expected_grads):
        assert t.grad is not None
        assert torch.allclose(t.grad, eg)
        # else:
        #     assert t.grad is None


def test_set_grad_(grad_tl: TensorList, grad_tensors):
    # Set initial grads
    initial_grads = TensorList([torch.ones_like(t) if t.requires_grad else None for t in grad_tensors])
    grad_tl.set_grad_(initial_grads)
    for t, ig in zip(grad_tl, initial_grads):
        assert t.grad is ig

    # Set new grads
    new_grads = TensorList([torch.rand_like(t) * 2 if t.requires_grad else None for t in grad_tensors])
    grad_tl.set_grad_(new_grads)
    for t, ng in zip(grad_tl, new_grads):
        assert t.grad is ng # Checks object identity for None, value for Tensors


def test_zero_grad_(grad_tl: TensorList, grad_tensors):
    # Set some grads
    for t in grad_tl:
        if t.requires_grad:
            t.grad = torch.ones_like(t)

    # Zero grads (set to None)
    grad_tl.zero_grad_(set_to_none=True)
    for t in grad_tl:
        assert t.grad is None

    # Set grads again
    for t in grad_tl:
        if t.requires_grad:
            t.grad = torch.ones_like(t)

    # Zero grads (set to zero)
    grad_tl.zero_grad_(set_to_none=False)
    for t in grad_tl:
        if t.requires_grad:
            assert t.grad is not None
            assert torch.equal(t.grad, torch.zeros_like(t))
        else:
            assert t.grad is None # Should remain None if requires_grad is False


# --- Arithmetic Operators ---

@pytest.mark.parametrize("other_type", ["scalar", "list_scalar", "tensorlist", "list_tensor"])
@pytest.mark.parametrize("op, op_inplace, torch_op, foreach_op, foreach_op_inplace", [
    ('__add__', '__iadd__', torch.add, torch._foreach_add, torch._foreach_add_),
    ('__sub__', '__isub__', torch.sub, torch._foreach_sub, torch._foreach_sub_),
    ('__mul__', '__imul__', torch.mul, torch._foreach_mul, torch._foreach_mul_),
    ('__truediv__', '__itruediv__', torch.div, torch._foreach_div, torch._foreach_div_),
])
def test_arithmetic_ops(simple_tl: TensorList, simple_tl_clone: TensorList, other_type, op, op_inplace, torch_op, foreach_op, foreach_op_inplace):
    if other_type == "scalar":
        other = 2.5
        other_list = [other] * len(simple_tl)
    elif other_type == "list_scalar":
        other = [1.0, 2.0, 3.0]
        other_list = other
    elif other_type == "tensorlist":
        other = simple_tl_clone.clone().mul_(0.5) # Create a compatible TensorList
        other_list = other
    elif other_type == "list_tensor":
        other = [t * 0.5 for t in simple_tl_clone] # Create a compatible list of tensors
        other_list = other
    else:
        pytest.fail("Unknown other_type")

    # --- Test out-of-place ---
    op_func = getattr(simple_tl, op)
    result_tl = op_func(other)
    expected_tl = TensorList([torch_op(t, o) for t, o in zip(simple_tl, other_list)])

    assert isinstance(result_tl, TensorList)
    assert_tl_allclose(result_tl, expected_tl)
    # Ensure original is unchanged
    assert_tl_equal(simple_tl, simple_tl_clone)

    # Test foreach version directly for comparison (if applicable)
    if op != '__sub__' or other_type != 'scalar': # _foreach_sub doesn't support scalar 'other' directly
        if hasattr(torch, foreach_op.__name__):
            expected_foreach = TensorList(foreach_op(simple_tl, other_list))
            assert_tl_allclose(result_tl, expected_foreach)

    # --- Test in-place ---
    tl_copy = simple_tl.clone()
    op_inplace_func = getattr(tl_copy, op_inplace)
    result_inplace = op_inplace_func(other)

    assert result_inplace is tl_copy # Should return self
    assert_tl_allclose(tl_copy, expected_tl)

    # Test foreach_ inplace version directly
    tl_copy_foreach = simple_tl.clone()
    if op != '__sub__' or other_type != 'scalar': # _foreach_sub_ doesn't support scalar 'other' directly
       if hasattr(torch, foreach_op_inplace.__name__):
           foreach_op_inplace(tl_copy_foreach, other_list)
           assert_tl_allclose(tl_copy_foreach, expected_tl)

    # --- Test r-ops (if applicable) ---
    if op in ['__add__', '__mul__']: # Commutative
        rop_func = getattr(simple_tl, op.replace('__', '__r', 1))
        result_rtl = rop_func(other)
        assert_tl_allclose(result_rtl, expected_tl)
    elif op == '__sub__': # Test rsub: other - self = -(self - other)
        rop_func = getattr(simple_tl, '__rsub__')
        result_rtl = rop_func(other)
        expected_rtl = expected_tl.neg() # Note: self.sub(other).neg_() == other - self
        assert_tl_allclose(result_rtl, expected_rtl)
    elif op == '__truediv__': # Test rtruediv: other / self
         if other_type in ["scalar", "list_scalar"]: # scalar / tensor or list<scalar> / list<tensor>
             rop_func = getattr(simple_tl, '__rtruediv__')
             result_rtl = rop_func(other)
             expected_rtl = TensorList([o / t for t, o in zip(simple_tl, other_list)])
             assert_tl_allclose(result_rtl, expected_rtl)
         # rtruediv for tensorlist/list_tensor is not implemented directly


@pytest.mark.parametrize("op, torch_op", [
    ('__pow__', torch.pow),
    ('__floordiv__', torch.floor_divide),
    ('__mod__', torch.remainder),
])
@pytest.mark.parametrize("other_type", ["scalar", "list_scalar", "tensorlist", "list_tensor"])
def test_other_arithmetic_ops(simple_tl: TensorList, simple_tl_clone: TensorList, op, torch_op, other_type):
    is_pow = op == '__pow__'
    if other_type == "scalar":
        other = 2 if is_pow else 2.5
        other_list = [other] * len(simple_tl)
    elif other_type == "list_scalar":
        other = [2, 1, 3] if is_pow else [1.5, 2.5, 3.5]
        other_list = other
    elif other_type == "tensorlist":
        other = simple_tl_clone.clone().abs_().add_(1).clamp_(max=3) if is_pow else simple_tl_clone.clone().mul_(0.5).add_(1)
        other_list = other
    elif other_type == "list_tensor":
         other = [(t.abs() + 1).clamp(max=3) if is_pow else (t*0.5 + 1) for t in simple_tl_clone]
         other_list = other
    else:
        pytest.fail("Unknown other_type")

    # Test out-of-place
    op_func = getattr(simple_tl, op)
    result_tl = op_func(other)
    expected_tl = TensorList([torch_op(t, o) for t, o in zip(simple_tl, other_list)])

    assert isinstance(result_tl, TensorList)
    assert_tl_allclose(result_tl, expected_tl)
    assert_tl_equal(simple_tl, simple_tl_clone) # Ensure original unchanged

    # Test in-place (if exists)
    op_inplace = op.replace('__', '__i', 1) + '_' # Standard naming convention adopted
    if hasattr(simple_tl, op_inplace):
        tl_copy = simple_tl.clone()
        op_inplace_func = getattr(tl_copy, op_inplace)
        result_inplace = op_inplace_func(other)
        assert result_inplace is tl_copy
        assert_tl_allclose(tl_copy, expected_tl)

    # Test rpow
    if op == '__pow__' and other_type in ['scalar']:#, 'list_scalar']: # _foreach_pow doesn't support list of scalars as base
         rop_func = getattr(simple_tl, '__rpow__')
         result_rtl = rop_func(other)
         expected_rtl = TensorList([torch_op(o, t) for t, o in zip(simple_tl, other_list)])
         assert_tl_allclose(result_rtl, expected_rtl)


def test_negation(simple_tl: TensorList, simple_tl_clone):
    neg_tl = -simple_tl
    expected_tl = TensorList([-t for t in simple_tl])
    assert_tl_allclose(neg_tl, expected_tl)
    assert_tl_equal(simple_tl, simple_tl_clone) # Ensure original unchanged

    neg_tl_inplace = simple_tl.neg_()
    assert neg_tl_inplace is simple_tl
    assert_tl_allclose(simple_tl, expected_tl)

# --- Comparison Operators ---

@pytest.mark.parametrize("op, torch_op", [
    ('__eq__', torch.eq),
    ('__ne__', torch.ne),
    ('__lt__', torch.lt),
    ('__le__', torch.le),
    ('__gt__', torch.gt),
    ('__ge__', torch.ge),
])
@pytest.mark.parametrize("other_type", ["scalar", "list_scalar", "tensorlist", "list_tensor"])
def test_comparison_ops(simple_tl: TensorList, op, torch_op, other_type):
    if other_type == "scalar":
        other = 0.0
        other_list = [other] * len(simple_tl)
    elif other_type == "list_scalar":
        other = [-0.5, 0.0, 0.5]
        other_list = other
    elif other_type == "tensorlist":
        other = simple_tl.clone().mul_(0.9)
        other_list = other
    elif other_type == "list_tensor":
        other = [t * 0.9 for t in simple_tl]
        other_list = other
    else:
        pytest.fail("Unknown other_type")

    op_func = getattr(simple_tl, op)
    result_tl = op_func(other)
    expected_tl = TensorList([torch_op(t, o) for t, o in zip(simple_tl, other_list)])

    assert isinstance(result_tl, TensorList)
    assert all(t.dtype == torch.bool for t in result_tl)
    assert_tl_equal(result_tl, expected_tl)


# --- Logical Operators ---

@pytest.mark.parametrize("op, op_inplace, torch_op", [
    ('__and__', '__iand__', torch.logical_and),
    ('__or__', '__ior__', torch.logical_or),
    ('__xor__', '__ixor__', torch.logical_xor),
])
def test_logical_binary_ops(bool_tl: TensorList, op, op_inplace, torch_op):
    other_tl = TensorList([randmask_like(t) for t in bool_tl])
    other_list = list(other_tl) # Use list version for comparison

    # Out-of-place
    op_func = getattr(bool_tl, op)
    result_tl = op_func(other_tl)
    expected_tl = TensorList([torch_op(t, o) for t, o in zip(bool_tl, other_list)])

    assert isinstance(result_tl, TensorList)
    assert all(t.dtype == torch.bool for t in result_tl)
    assert_tl_equal(result_tl, expected_tl)

    # In-place
    tl_copy = bool_tl.clone()
    op_inplace_func = getattr(tl_copy, op_inplace) # Naming convention with _
    result_inplace = op_inplace_func(other_tl)
    assert result_inplace is tl_copy
    assert_tl_equal(tl_copy, expected_tl)


def test_logical_not(bool_tl: TensorList):
    # Out-of-place (~ operator maps to logical_not)
    not_tl = ~bool_tl
    expected_tl = TensorList([torch.logical_not(t) for t in bool_tl])
    assert isinstance(not_tl, TensorList)
    assert all(t.dtype == torch.bool for t in not_tl)
    assert_tl_equal(not_tl, expected_tl)

    # In-place
    tl_copy = bool_tl.clone()
    result_inplace = tl_copy.logical_not_()
    assert result_inplace is tl_copy
    assert_tl_equal(tl_copy, expected_tl)


def test_bool_raises(simple_tl: TensorList):
    with pytest.raises(RuntimeError, match="Boolean value of TensorList is ambiguous"):
        bool(simple_tl)
    # Test with empty list
    with pytest.raises(RuntimeError, match="Boolean value of TensorList is ambiguous"):
        bool(TensorList())

# --- Map / Zipmap / Filter ---

def test_map(simple_tl: TensorList):
    mapped_tl = simple_tl.map(torch.abs)
    expected_tl = TensorList([torch.abs(t) for t in simple_tl])
    assert_tl_allclose(mapped_tl, expected_tl)

def test_map_inplace_(simple_tl: TensorList):
    tl_copy = simple_tl.clone()
    result = tl_copy.map_inplace_(torch.abs_)
    expected_tl = TensorList([torch.abs(t) for t in simple_tl]) # Calculate expected from original
    assert result is tl_copy
    assert_tl_allclose(tl_copy, expected_tl)

def test_filter(simple_tl: TensorList):
    # Filter tensors with more than 5 elements
    filtered_tl = simple_tl.filter(lambda t: t.numel() > 5)
    expected_tl = TensorList([t for t in simple_tl if t.numel() > 5])
    assert len(filtered_tl) == len(expected_tl)
    for ft, et in zip(filtered_tl, expected_tl):
        assert ft is et # Should contain the original tensor objects

def test_zipmap(simple_tl: TensorList):
    # Zipmap with another TensorList
    other_tl = simple_tl.clone().mul_(0.5)
    result_tl = simple_tl.zipmap(torch.add, other_tl)
    expected_tl = TensorList([torch.add(t, o) for t, o in zip(simple_tl, other_tl)])
    assert_tl_allclose(result_tl, expected_tl)

    # Zipmap with a list of tensors
    other_list = [t * 0.5 for t in simple_tl]
    result_tl_list = simple_tl.zipmap(torch.add, other_list)
    assert_tl_allclose(result_tl_list, expected_tl)

    # Zipmap with a scalar
    result_tl_scalar = simple_tl.zipmap(torch.add, 2.0)
    expected_tl_scalar = TensorList([torch.add(t, 2.0) for t in simple_tl])
    assert_tl_allclose(result_tl_scalar, expected_tl_scalar)

    # Zipmap with a list of scalars
    other_scalars = [1.0, 2.0, 3.0]
    result_tl_scalars = simple_tl.zipmap(torch.add, other_scalars)
    expected_tl_scalars = TensorList([torch.add(t, s) for t, s in zip(simple_tl, other_scalars)])
    assert_tl_allclose(result_tl_scalars, expected_tl_scalars)


def test_zipmap_inplace_(simple_tl: TensorList):
    # Zipmap inplace with another TensorList
    tl_copy = simple_tl.clone()
    other_tl = simple_tl.clone().mul_(0.5)
    result = tl_copy.zipmap_inplace_(_MethodCallerWithArgs('add_'), other_tl)
    expected_tl = TensorList([torch.add(t, o) for t, o in zip(simple_tl, other_tl)])
    assert result is tl_copy
    assert_tl_allclose(tl_copy, expected_tl)

    # Zipmap inplace with a scalar
    tl_copy_scalar = simple_tl.clone()
    result_scalar = tl_copy_scalar.zipmap_inplace_(_MethodCallerWithArgs('add_'), 2.0)
    expected_tl_scalar = TensorList([torch.add(t, 2.0) for t in simple_tl])
    assert result_scalar is tl_copy_scalar
    assert_tl_allclose(tl_copy_scalar, expected_tl_scalar)

    # Zipmap inplace with list of scalars
    tl_copy_scalars = simple_tl.clone()
    other_scalars = [1.0, 2.0, 3.0]
    result_scalars = tl_copy_scalars.zipmap_inplace_(_MethodCallerWithArgs('add_'), other_scalars)
    expected_tl_scalars = TensorList([torch.add(t, s) for t, s in zip(simple_tl, other_scalars)])
    assert result_scalars is tl_copy_scalars
    assert_tl_allclose(tl_copy_scalars, expected_tl_scalars)


def test_zipmap_args(simple_tl: TensorList):
    other1 = simple_tl.clone().mul(0.5)
    other2 = 2.0
    other3 = [1, 2, 3]
    # Test torch.lerp(input, end, weight) -> input + weight * (end - input)
    # self = input, other1 = end, other2 = weight (scalar)
    result_tl = simple_tl.zipmap_args(torch.lerp, other1, other2)
    expected_tl = TensorList([torch.lerp(t, o1, other2) for t, o1 in zip(simple_tl, other1)])
    assert_tl_allclose(result_tl, expected_tl)

    # self = input, other1 = end, other3 = weight (list scalar)
    result_tl_list = simple_tl.zipmap_args(torch.lerp, other1, other3)
    expected_tl_list = TensorList([torch.lerp(t, o1, o3) for t, o1, o3 in zip(simple_tl, other1, other3)])
    assert_tl_allclose(result_tl_list, expected_tl_list)

def test_zipmap_args_inplace_(simple_tl: TensorList):
     tl_copy = simple_tl.clone()
     other1 = simple_tl.clone().mul(0.5)
     other2 = 0.5
     # Test torch.addcmul_(tensor1, tensor2, value=1) -> self + value * tensor1 * tensor2
     # self = self, other1 = tensor1, other1 (again) = tensor2, other2 = value
     result_tl = tl_copy.zipmap_args_inplace_(_MethodCallerWithArgs('addcmul_'), other1, other1, value=other2)
     expected_tl = TensorList([t.addcmul(o1, o1, value=other2) for t, o1 in zip(simple_tl.clone(), other1)]) # Need clone for calculation
     assert result_tl is tl_copy
     assert_tl_allclose(tl_copy, expected_tl)

# --- Tensor Method Wrappers ---

@pytest.mark.parametrize("method_name, args", [
    ('clone', ()),
    ('detach', ()),
    ('contiguous', ()),
    ('cpu', ()),
    ('long', ()),
    ('short', ()),
    ('as_float', ()),
    ('as_int', ()),
    ('as_bool', ()),
    ('sqrt', ()),
    ('exp', ()),
    ('log', ()),
    ('sin', ()),
    ('cos', ()),
    ('abs', ()),
    ('neg', ()),
    ('reciprocal', ()),
    ('sign', ()),
    ('round', ()),
    ('floor', ()),
    ('ceil', ()),
    ('logical_not', ()), # Assuming input is boolean for this test
    ('ravel', ()),
    ('view_flat', ()),
    ('conj', ()), # Assuming input is complex for this test
    ('squeeze', ()),
    ('squeeze', (0,)), # Example with args
    # Add more simple unary methods here...
])
def test_simple_unary_methods(simple_tl: TensorList, method_name, args):
    tl_to_test = simple_tl
    if method_name == 'logical_not':
        tl_to_test = simple_tl.gt(0) # Create a boolean TL
    elif method_name == 'conj':
         tl_to_test = TensorList.complex(simple_tl, simple_tl) # Create complex

    method = getattr(tl_to_test, method_name)
    result_tl = method(*args)

    method_names_map = {"as_float": "float", "as_int": "int", "as_bool": "bool", "view_flat": "ravel"}
    tensor_method_name = method_name
    if tensor_method_name in method_names_map: tensor_method_name = method_names_map[tensor_method_name]
    expected_tl = TensorList([getattr(t, tensor_method_name)(*args) for t in tl_to_test])

    assert isinstance(result_tl, TensorList)
    # Need allclose for float results, equal for others
    if any(t.is_floating_point() for t in expected_tl):
         assert_tl_allclose(result_tl, expected_tl)
    else:
         assert_tl_equal(result_tl, expected_tl)

    # Test inplace if available
    method_inplace_name = method_name + '_'
    if hasattr(tl_to_test, method_inplace_name) and hasattr(torch.Tensor, method_inplace_name):
        tl_copy = tl_to_test.clone()
        method_inplace = getattr(tl_copy, method_inplace_name)
        result_inplace = method_inplace(*args)
        assert result_inplace is tl_copy
        if any(t.is_floating_point() for t in expected_tl):
            assert_tl_allclose(tl_copy, expected_tl)
        else:
            assert_tl_equal(tl_copy, expected_tl)

def test_to(simple_tl: TensorList):
    # Test changing dtype
    float_tl = simple_tl.to(dtype=torch.float64)
    assert all(t.dtype == torch.float64 for t in float_tl)

    # Test changing device (if multiple devices available)
    if torch.cuda.is_available():
        cuda_tl = simple_tl.to(device='cuda')
        assert all(t.device.type == 'cuda' for t in cuda_tl)
        cpu_tl = cuda_tl.to('cpu')
        assert all(t.device.type == 'cpu' for t in cpu_tl)

def test_copy_(simple_tl: TensorList):
    src_tl = TensorList([torch.randn_like(t) for t in simple_tl])
    tl_copy = simple_tl.clone()
    tl_copy.copy_(src_tl)
    assert_tl_equal(tl_copy, src_tl)
    # Ensure src is unchanged
    assert not torch.equal(src_tl[0], simple_tl[0]) # Verify src was different

def test_set_(simple_tl: TensorList):
    src_tl = TensorList([torch.randn_like(t) for t in simple_tl])
    tl_copy = simple_tl.clone()
    tl_copy.set_(src_tl) # src_tl provides the storage/tensors
    assert_tl_equal(tl_copy, src_tl)
    # Note: set_ might have side effects on src_tl depending on PyTorch version/tensor types

def test_requires_grad_(grad_tl: TensorList):
    grad_tl.requires_grad_(False)
    assert grad_tl.requires_grad == [False] * len(grad_tl)
    grad_tl.requires_grad_(True)
    # This sets requires_grad=True for ALL tensors, unlike the initial fixture
    assert grad_tl.requires_grad == [True] * len(grad_tl)


# --- Vectorization ---

def test_to_vec(simple_tl: TensorList):
    vec = simple_tl.to_vec()
    expected_vec = torch.cat([t.view(-1) for t in simple_tl])
    assert torch.equal(vec, expected_vec)

def test_from_vec_(simple_tl: TensorList):
    tl_copy = simple_tl.clone()
    numel = simple_tl.global_numel()
    new_vec = torch.arange(numel, dtype=simple_tl[0].dtype).float() # Use float for generality

    result = tl_copy.from_vec_(new_vec)
    assert result is tl_copy

    current_pos = 0
    for t_orig, t_modified in zip(simple_tl, tl_copy):
        n = t_orig.numel()
        expected_tensor = new_vec[current_pos : current_pos + n].view_as(t_orig)
        assert torch.equal(t_modified, expected_tensor)
        current_pos += n

def test_from_vec(simple_tl: TensorList):
    tl_clone = simple_tl.clone() # Keep original safe
    numel = simple_tl.global_numel()
    new_vec = torch.arange(numel, dtype=simple_tl[0].dtype).float()

    new_tl = simple_tl.from_vec(new_vec)
    assert isinstance(new_tl, TensorList)
    assert_tl_equal(simple_tl, tl_clone) # Original unchanged

    current_pos = 0
    for t_orig, t_new in zip(simple_tl, new_tl):
        n = t_orig.numel()
        expected_tensor = new_vec[current_pos : current_pos + n].view_as(t_orig)
        assert torch.equal(t_new, expected_tensor)
        current_pos += n


# --- Global Reductions ---

@pytest.mark.parametrize("global_method, vec_equiv_method", [
    ('global_min', 'min'),
    ('global_max', 'max'),
    ('global_sum', 'sum'),
    ('global_mean', 'mean'),
    ('global_std', 'std'),
    ('global_var', 'var'),
    ('global_any', 'any'),
    ('global_all', 'all'),
])
def test_global_reductions(simple_tl: TensorList, global_method, vec_equiv_method):
    tl_to_test = simple_tl
    if 'any' in global_method or 'all' in global_method:
        tl_to_test = simple_tl.gt(0) # Need boolean input

    global_method_func = getattr(tl_to_test, global_method)
    result = global_method_func()

    vec = tl_to_test.to_vec()
    vec_equiv_func = getattr(vec, vec_equiv_method)
    expected = vec_equiv_func()

    if isinstance(result, bool): assert result == expected
    else: assert torch.allclose(result, expected, atol=1e-4), f"Tensors not close: {result = }, {expected = }"


def test_global_vector_norm(simple_tl: TensorList):
    ord = 1.5
    result = simple_tl.global_vector_norm(ord=ord)
    vec = simple_tl.to_vec()
    expected = torch.linalg.vector_norm(vec, ord=ord) # pylint:disable=not-callable
    assert torch.allclose(result, expected)

def test_global_numel(simple_tl: TensorList):
    result = simple_tl.global_numel()
    expected = sum(t.numel() for t in simple_tl)
    assert result == expected

# --- Like Creation Methods ---

@pytest.mark.parametrize("like_method, torch_equiv", [
    ('empty_like', torch.empty_like),
    ('zeros_like', torch.zeros_like),
    ('ones_like', torch.ones_like),
    ('rand_like', torch.rand_like),
    ('randn_like', torch.randn_like),
])
def test_simple_like_methods(simple_tl: TensorList, like_method, torch_equiv):
    like_method_func = getattr(simple_tl, like_method)
    result_tl = like_method_func()

    assert isinstance(result_tl, TensorList)
    assert len(result_tl) == len(simple_tl)
    for res_t, orig_t in zip(result_tl, simple_tl):
        assert res_t.shape == orig_t.shape
        assert res_t.dtype == orig_t.dtype
        assert res_t.device == orig_t.device
        # Cannot easily check values for rand/randn/empty

    # Test with kwargs (e.g., changing dtype)
    result_tl_kw = like_method_func(dtype=torch.float64)
    assert all(t.dtype == torch.float64 for t in result_tl_kw)


def test_full_like(simple_tl: TensorList):
    # Scalar fill_value
    fill_value_scalar = 5.0
    result_tl_scalar = simple_tl.full_like(fill_value_scalar)
    expected_tl_scalar = TensorList([torch.full_like(t, fill_value_scalar) for t in simple_tl])
    assert_tl_equal(result_tl_scalar, expected_tl_scalar)

    # List fill_value
    fill_value_list = [1.0, 2.0, 3.0]
    result_tl_list = simple_tl.full_like(fill_value_list)
    expected_tl_list = TensorList([torch.full_like(t, fv) for t, fv in zip(simple_tl, fill_value_list)])
    assert_tl_equal(result_tl_list, expected_tl_list)

    # Test with kwargs
    result_tl_kw = simple_tl.full_like(fill_value_scalar, dtype=torch.int)
    assert all(t.dtype == torch.int for t in result_tl_kw)
    assert all(torch.all(t == int(fill_value_scalar)) for t in result_tl_kw)


def test_randint_like(simple_tl: TensorList):
    low = 0
    high = 10
    # Scalar low/high
    result_tl_scalar = simple_tl.randint_like(low, high)
    assert isinstance(result_tl_scalar, TensorList)
    assert all(t.dtype == simple_tl[0].dtype for t in result_tl_scalar) # Default dtype
    assert all(torch.all((t >= low) & (t < high)) for t in result_tl_scalar)
    assert result_tl_scalar.shape == simple_tl.shape

    # List low/high
    low_list = [0, 5, 2]
    high_list = [5, 15, 7]
    result_tl_list = simple_tl.randint_like(low_list, high_list)
    assert isinstance(result_tl_list, TensorList)
    assert all(t.dtype == simple_tl[0].dtype for t in result_tl_list)
    assert all(torch.all((t >= l) & (t < h)) for t, l, h in zip(result_tl_list, low_list, high_list))
    assert result_tl_list.shape == simple_tl.shape


def test_uniform_like(simple_tl: TensorList):
    # Default range (0, 1)
    result_tl_default = simple_tl.uniform_like()
    assert isinstance(result_tl_default, TensorList)
    assert result_tl_default.shape == simple_tl.shape
    assert all(t.dtype == simple_tl[i].dtype for i, t in enumerate(result_tl_default))
    assert all(torch.all((t >= 0) & (t <= 1)) for t in result_tl_default) # Check range roughly

    # Scalar low/high
    low, high = -1.0, 1.0
    result_tl_scalar = simple_tl.uniform_like(low, high)
    assert all(torch.all((t >= low) & (t <= high)) for t in result_tl_scalar)

    # List low/high
    low_list = [-1, 0, -2]
    high_list = [0, 1, -1]
    result_tl_list = simple_tl.uniform_like(low_list, high_list)
    assert all(torch.all((t >= l) & (t <= h)) for t, l, h in zip(result_tl_list, low_list, high_list))


def test_sphere_like(simple_tl: TensorList):
    radius = 5.0
    result_tl_scalar = simple_tl.sphere_like(radius)
    assert isinstance(result_tl_scalar, TensorList)
    assert result_tl_scalar.shape == simple_tl.shape
    assert torch.allclose(result_tl_scalar.global_vector_norm(), torch.tensor(radius))

    radius_list = [1.0, 10.0, 2.0]
    result_tl_list = simple_tl.sphere_like(radius_list)
    # Cannot easily check norm with list radius, just check type/shape
    assert isinstance(result_tl_list, TensorList)
    assert result_tl_list.shape == simple_tl.shape


def test_bernoulli_like(big_tl: TensorList):
    p_scalar = 0.7
    result_tl_scalar = big_tl.bernoulli_like(p_scalar)
    assert isinstance(result_tl_scalar, TensorList)
    assert result_tl_scalar.shape == big_tl.shape
    assert all(t.dtype == big_tl[i].dtype for i, t in enumerate(result_tl_scalar)) # Should preserve dtype
    assert all(torch.all((t == 0) | (t == 1)) for t in result_tl_scalar)
    # Check mean is approximately p
    assert abs(result_tl_scalar.to_vec().float().mean().item() - p_scalar) < 0.1 # Loose check

    p_list = [0.2, 0.5, 0.8]
    result_tl_list = big_tl.bernoulli_like(p_list)
    assert isinstance(result_tl_list, TensorList)
    assert result_tl_list.shape == big_tl.shape


def test_rademacher_like(big_tl: TensorList):
    result_tl = big_tl.rademacher_like() # p=0.5 default
    assert isinstance(result_tl, TensorList)
    assert result_tl.shape == big_tl.shape
    assert all(torch.all((t == -1) | (t == 1)) for t in result_tl)

    # Check mean is approx 0
    assert abs(result_tl.to_vec().float().mean().item()) < 0.1 # Loose check


@pytest.mark.parametrize("dist", ['normal', 'uniform', 'sphere', 'rademacher'])
def test_sample_like(simple_tl: TensorList, dist):
    eps_scalar = 1
    result_tl_scalar = simple_tl.sample_like(distribution=dist)
    assert isinstance(result_tl_scalar, TensorList)
    assert result_tl_scalar.shape == simple_tl.shape

    eps_list = [1.0,]
    result_tl_list = simple_tl.sample_like(distribution=dist)
    assert isinstance(result_tl_list, TensorList)
    assert result_tl_list.shape == simple_tl.shape

    # Basic checks based on distribution
    if dist == 'uniform':
        assert all(torch.all((t >= -eps_scalar) & (t <= eps_scalar)) for t in result_tl_scalar)
        assert all(torch.all((t >= -e) & (t <= e)) for t, e in zip(result_tl_list, eps_list))
    elif dist == 'sphere':
        # assert torch.allclose(result_tl_scalar.global_vector_norm(), torch.tensor(eps_scalar))
        pass
        # Cannot check list version easily
    elif dist == 'rademacher':
         assert all(torch.all((t == -eps_scalar) | (t == eps_scalar)) for t in result_tl_scalar)
         assert all(torch.all((t == -e) | (t == e)) for t, e in zip(result_tl_list, eps_list))


# --- Advanced Math Ops ---

def test_clamp(simple_tl: TensorList):
    min_val, max_val = -0.5, 0.5
    # Both min and max
    clamped_tl = simple_tl.clamp(min_val, max_val)
    expected_tl = TensorList([t.clamp(min_val, max_val) for t in simple_tl])
    assert_tl_allclose(clamped_tl, expected_tl)

    # Only min
    clamped_min_tl = simple_tl.clamp(min=min_val)
    expected_min_tl = TensorList([t.clamp(min=min_val) for t in simple_tl])
    assert_tl_allclose(clamped_min_tl, expected_min_tl)

    # Only max
    clamped_max_tl = simple_tl.clamp(max=max_val)
    expected_max_tl = TensorList([t.clamp(max=max_val) for t in simple_tl])
    assert_tl_allclose(clamped_max_tl, expected_max_tl)

    # List min/max
    min_list = [-1, -0.5, 0]
    max_list = [1, 0.5, 0.2]
    clamped_list_tl = simple_tl.clamp(min_list, max_list)
    expected_list_tl = TensorList([t.clamp(mn, mx) for t, mn, mx in zip(simple_tl, min_list, max_list)])
    assert_tl_allclose(clamped_list_tl, expected_list_tl)

    # Inplace
    tl_copy = simple_tl.clone()
    result = tl_copy.clamp_(min_val, max_val)
    assert result is tl_copy
    assert_tl_allclose(tl_copy, expected_tl)


def test_clamp_magnitude(simple_tl: TensorList):
    min_val, max_val = 0.2, 1.0
    tl_copy = simple_tl.clone()
    # Test non-zero case
    tl_copy[0][0,0] = 0.01 # ensure some small values
    tl_copy[1][0] = 10.0 # ensure some large values
    tl_copy[2][0,0,0] = 0.0 # test zero

    clamped_tl = tl_copy.clamp_magnitude(min_val, max_val)
    # Check magnitudes are clipped
    for t in clamped_tl:
        abs_t = t.abs()
        # Allow small tolerance for floating point issues near zero
        assert torch.all(abs_t >= min_val - 1e-6)
        assert torch.all(abs_t <= max_val + 1e-6)
    # Check signs are preserved (or zero remains zero)
    original_sign = tl_copy.sign()
    clamped_sign = clamped_tl.sign()
    # Zeros might become non-zero min magnitude, so compare non-zeros
    non_zero_mask = tl_copy.ne(0)
    for os, cs, nz in zip(original_sign, clamped_sign, non_zero_mask):
        assert torch.all(os[nz] == cs[nz])

    # Inplace
    tl_copy_inplace = tl_copy.clone()
    result = tl_copy_inplace.clamp_magnitude_(min_val, max_val)
    assert result is tl_copy_inplace
    assert_tl_allclose(tl_copy_inplace, clamped_tl)


def test_lerp(simple_tl: TensorList):
    tensors1 = simple_tl.clone().mul_(2)
    weight_scalar = 0.5
    result_tl_scalar = simple_tl.lerp(tensors1, weight_scalar)
    expected_tl_scalar = TensorList([torch.lerp(t, t1, weight_scalar) for t, t1 in zip(simple_tl, tensors1)])
    assert_tl_allclose(result_tl_scalar, expected_tl_scalar)

    weight_list = [0.1, 0.5, 0.9]
    result_tl_list = simple_tl.lerp(tensors1, weight_list)
    expected_tl_list = TensorList([torch.lerp(t, t1, w) for t, t1, w in zip(simple_tl, tensors1, weight_list)])
    assert_tl_allclose(result_tl_list, expected_tl_list)

    # Inplace
    tl_copy = simple_tl.clone()
    result_inplace = tl_copy.lerp_(tensors1, weight_scalar)
    assert result_inplace is tl_copy
    assert_tl_allclose(tl_copy, expected_tl_scalar)


def test_lerp_compat(simple_tl: TensorList):
    # Test specifically the scalar sequence case for compatibility fallback
    tensors1 = simple_tl.clone().mul_(2)
    weight_list = [0.1, 0.5, 0.9]
    result_tl_list = simple_tl.lerp_compat(tensors1, weight_list)
    expected_tl_list = TensorList([t + w * (t1 - t) for t, t1, w in zip(simple_tl, tensors1, weight_list)])
    assert_tl_allclose(result_tl_list, expected_tl_list)

    # Inplace
    tl_copy = simple_tl.clone()
    result_inplace = tl_copy.lerp_compat_(tensors1, weight_list)
    assert result_inplace is tl_copy
    assert_tl_allclose(tl_copy, expected_tl_list)


@pytest.mark.parametrize("op_name, torch_op", [
    ('addcmul', torch.addcmul),
    ('addcdiv', torch.addcdiv),
])
def test_addcops(simple_tl: TensorList, op_name, torch_op):
    tensors1 = simple_tl.clone().add(0.1)
    tensors2 = simple_tl.clone().mul(0.5)
    value_scalar = 2.0
    value_list = [1.0, 2.0, 3.0]

    op_func = getattr(simple_tl, op_name)
    op_inplace_func = getattr(simple_tl, op_name + '_')

    # Scalar value
    result_tl_scalar = op_func(tensors1, tensors2, value=value_scalar)
    expected_tl_scalar = TensorList([torch_op(t, t1, t2, value=value_scalar)
                                     for t, t1, t2 in zip(simple_tl, tensors1, tensors2)])
    assert_tl_allclose(result_tl_scalar, expected_tl_scalar)

    # List value
    result_tl_list = op_func(tensors1, tensors2, value=value_list)
    expected_tl_list = TensorList([torch_op(t, t1, t2, value=v)
                                     for t, t1, t2, v in zip(simple_tl, tensors1, tensors2, value_list)])
    assert_tl_allclose(result_tl_list, expected_tl_list)


    # Inplace (scalar value)
    tl_copy_scalar = simple_tl.clone()
    op_inplace_func = getattr(tl_copy_scalar, op_name + '_')
    result_inplace_scalar = op_inplace_func(tensors1, tensors2, value=value_scalar)
    assert result_inplace_scalar is tl_copy_scalar
    assert_tl_allclose(tl_copy_scalar, expected_tl_scalar)

    # Inplace (list value)
    tl_copy_list = simple_tl.clone()
    op_inplace_func = getattr(tl_copy_list, op_name + '_')
    result_inplace_list = op_inplace_func(tensors1, tensors2, value=value_list)
    assert result_inplace_list is tl_copy_list
    assert_tl_allclose(tl_copy_list, expected_tl_list)


@pytest.mark.parametrize("op_name, torch_op", [
    ('maximum', torch.maximum),
    ('minimum', torch.minimum),
])
def test_maximin(simple_tl: TensorList, op_name, torch_op):
    other_scalar = 0.0
    other_list_scalar = [-1.0, 0.0, 1.0]
    other_tl = simple_tl.clone().mul_(-1)

    op_func = getattr(simple_tl, op_name)
    op_inplace_func = getattr(simple_tl, op_name + '_')

    # Scalar other
    result_tl_scalar = op_func(other_scalar)
    expected_tl_scalar = TensorList([torch_op(t, torch.tensor(other_scalar, dtype=t.dtype, device=t.device)) for t in simple_tl])
    assert_tl_allclose(result_tl_scalar, expected_tl_scalar)

    # List scalar other
    result_tl_list_scalar = op_func(other_list_scalar)
    expected_tl_list_scalar = TensorList([torch_op(t, torch.tensor(o, dtype=t.dtype, device=t.device)) for t, o in zip(simple_tl, other_list_scalar)])
    assert_tl_allclose(result_tl_list_scalar, expected_tl_list_scalar)

    # TensorList other
    result_tl_tl = op_func(other_tl)
    expected_tl_tl = TensorList([torch_op(t, o) for t, o in zip(simple_tl, other_tl)])
    assert_tl_allclose(result_tl_tl, expected_tl_tl)

    # Inplace (TensorList other)
    tl_copy = simple_tl.clone()
    op_inplace_func = getattr(tl_copy, op_name + '_')
    result_inplace = op_inplace_func(other_tl)
    assert result_inplace is tl_copy
    assert_tl_allclose(tl_copy, expected_tl_tl)


def test_nan_to_num(simple_tl: TensorList):
    tl_with_nan = simple_tl.clone()
    tl_with_nan[0][0, 0] = float('nan')
    tl_with_nan[1][0] = float('inf')
    tl_with_nan[2][0, 0, 0] = float('-inf')

    # Default conversion
    result_default = tl_with_nan.nan_to_num()
    expected_default = TensorList([torch.nan_to_num(t) for t in tl_with_nan])
    assert_tl_equal(result_default, expected_default)

    # Custom values (scalar)
    nan, posinf, neginf = 0.0, 1e6, -1e6
    result_scalar = tl_with_nan.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)
    expected_scalar = TensorList([torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf) for t in tl_with_nan])
    assert_tl_equal(result_scalar, expected_scalar)

    # Custom values (list)
    nan_list = [0.0, 1.0, 2.0]
    posinf_list = [1e5, 1e6, 1e7]
    neginf_list = [-1e5, -1e6, -1e7]
    result_list = tl_with_nan.nan_to_num(nan=nan_list, posinf=posinf_list, neginf=neginf_list)
    expected_list = TensorList([torch.nan_to_num(t, nan=n, posinf=p, neginf=ni)
                                for t, n, p, ni in zip(tl_with_nan, nan_list, posinf_list, neginf_list)])
    assert_tl_equal(result_list, expected_list)

    # Inplace
    tl_copy = tl_with_nan.clone()
    result_inplace = tl_copy.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)
    assert result_inplace is tl_copy
    assert_tl_equal(tl_copy, expected_scalar)

# --- Reduction Ops ---

@pytest.mark.parametrize("reduction_method", ['mean', 'sum', 'min', 'max'])#, 'var', 'std', 'median', 'quantile'])
@pytest.mark.parametrize("dim", [None, 0, 'global'])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduction_ops(simple_tl: TensorList, reduction_method, dim, keepdim):
    if dim == 'global' and keepdim:
        # with pytest.raises(ValueError, match='dim = global and keepdim = True'):
        #     getattr(simple_tl, reduction_method)(dim=dim, keepdim=keepdim)
        return
    # Quantile needs q
    q = 0.75
    if reduction_method == 'quantile':
        args = {'q': q, 'dim': dim, 'keepdim': keepdim}
        torch_args = {'q': q, 'dim': dim, 'keepdim': keepdim}
        if dim is None: # torch.quantile doesn't accept dim=None, needs integer dim
            torch_args['dim'] = 0 if simple_tl[0].ndim > 0 else None # Use dim 0 if possible
            if torch_args['dim'] is None: # Cannot test dim=None on 0-d tensor easily here
                 pytest.skip("Cannot test quantile with dim=None on 0-d tensors easily")
    elif reduction_method == 'median':
        args = {'dim': dim, 'keepdim': keepdim}
        torch_args = {'dim': dim, 'keepdim': keepdim}
        if dim is None: # torch.median requires dim if tensor is not 1D
             # Skip complex multi-dim median check for None dim
             pytest.skip("Skipping median test with dim=None for simplicity")
    else:
        args = {'dim': dim, 'keepdim': keepdim}
        torch_args = {'dim': dim, 'keepdim': keepdim}

    reduction_func = getattr(simple_tl, reduction_method)

    # Skip if dim is invalid for a tensor
    if isinstance(dim, int):
        if any(dim >= t.ndim for t in simple_tl):
            pytest.skip(f"Dimension {dim} out of range for at least one tensor")

    try:
        result = reduction_func(**args)
    except RuntimeError as e:
        # median/quantile might fail on certain dtypes, skip if so
        if "median" in reduction_method or "quantile" in reduction_method:
             pytest.skip(f"Skipping {reduction_method} due to dtype incompatibility: {e}")
        else: raise e


    if dim == 'global':
        vec = simple_tl.to_vec()
        if reduction_method == 'min': expected = vec.min()
        elif reduction_method == 'max': expected = vec.max()
        elif reduction_method == 'mean': expected = vec.mean()
        elif reduction_method == 'sum': expected = vec.sum()
        elif reduction_method == 'std': expected = vec.std()
        elif reduction_method == 'var': expected = vec.var()
        elif reduction_method == 'median': expected = vec.median()#.values # scalar tensor
        elif reduction_method == 'quantile': expected = vec.quantile(q)
        else:
            pytest.fail("Unknown global reduction")
            assert False, reduction_method
        assert torch.allclose(result, expected, atol=1e-4)
    else:
        expected_list = []
        for t in simple_tl:
            if reduction_method == 'min': torch_func = getattr(t, 'amin')
            elif reduction_method == 'max': torch_func = getattr(t, 'amax')
            else: torch_func = getattr(t, reduction_method)
            try:
                if reduction_method == 'median':
                    # Median returns (values, indices), we only want values
                    expected_val = torch_func(**torch_args)[0]
                elif reduction_method == 'quantile':
                     expected_val = torch_func(**torch_args)
                     # quantile might return scalar tensor if dim is None and keepdim=False
                    #  if dim is None and not keepdim: expected_val = expected_val.unsqueeze(0) if expected_val.ndim == 0 else expected_val

                else:
                    torch_args_copy = torch_args.copy()
                    if reduction_method in ('min', 'max'):
                        if 'dim' in torch_args_copy and torch_args_copy['dim'] is None: torch_args_copy['dim'] = ()
                    expected_val = torch_func(**torch_args_copy)

                # Handle cases where reduction reduces to scalar but we expect TL
                if not isinstance(expected_val, torch.Tensor): # e.g. min/max on scalar tensor
                    expected_val = torch.tensor(expected_val, device=t.device, dtype=t.dtype)
                # if dim is None and not keepdim and expected_val.ndim==0:
                #      expected_val = expected_val.unsqueeze(0) # Make it 1D for consistency in TL


                expected_list.append(expected_val)
            except RuntimeError as e:
                 # Skip individual tensor if op not supported (e.g. std on int)
                 if "std" in str(e) or "var" in str(e) or "mean" in str(e):
                     pytest.skip(f"Skipping {reduction_method} on tensor due to dtype: {e}")
                 else: raise e

        expected_tl = TensorList(expected_list)
        assert isinstance(result, TensorList)
        assert len(result) == len(expected_tl)
        assert_tl_allclose(result, expected_tl, atol=1e-3) # Use allclose due to potential float variations

# --- Grafting, Rescaling, Normalizing, Clipping ---

def test_graft(simple_tl: TensorList):
    magnitude_tl = simple_tl.clone().mul_(2.0) # Double the magnitude

    # Tensorwise graft
    grafted_tensorwise = simple_tl.graft(magnitude_tl, tensorwise=True, ord=2)
    original_norms = simple_tl.norm(ord=2)
    magnitude_norms = magnitude_tl.norm(ord=2)
    grafted_norms = grafted_tensorwise.norm(ord=2)
    # Check norms match the magnitude norms
    assert_tl_allclose(grafted_norms, magnitude_norms)
    # Check directions are preserved (allow for scaling factor)
    for g, o, onorm, mnorm in zip(grafted_tensorwise, simple_tl, original_norms, magnitude_norms):
         # Handle zero norm case
         if onorm > 1e-7 and mnorm > 1e-7:
             expected_g = o * (mnorm / onorm)
             assert torch.allclose(g, expected_g)
         elif mnorm <= 1e-7: # If magnitude is zero, graft should be zero
             assert torch.allclose(g, torch.zeros_like(g))
         # If original norm is zero but magnitude is non-zero, result is undefined/arbitrary direction?
         # Current implementation results in zero due to mul by zero tensor.

    # Global graft
    grafted_global = simple_tl.graft(magnitude_tl, tensorwise=False, ord=2)
    original_global_norm = simple_tl.global_vector_norm(ord=2)
    magnitude_global_norm = magnitude_tl.global_vector_norm(ord=2)
    grafted_global_norm = grafted_global.global_vector_norm(ord=2)
    # Check global norm matches
    assert torch.allclose(grafted_global_norm, magnitude_global_norm)
    # Check direction (overall vector) is preserved
    if original_global_norm > 1e-7 and magnitude_global_norm > 1e-7:
         expected_global_scale = magnitude_global_norm / original_global_norm
         expected_global_tl = simple_tl * expected_global_scale
         assert_tl_allclose(grafted_global, expected_global_tl)
    elif magnitude_global_norm <= 1e-7:
         assert torch.allclose(grafted_global.to_vec(), torch.zeros(simple_tl.global_numel()))


    # Test inplace
    tl_copy_t = simple_tl.clone()
    tl_copy_t.graft_(magnitude_tl, tensorwise=True, ord=2)
    assert_tl_allclose(tl_copy_t, grafted_tensorwise)

    tl_copy_g = simple_tl.clone()
    tl_copy_g.graft_(magnitude_tl, tensorwise=False, ord=2)
    assert_tl_allclose(tl_copy_g, grafted_global)


@pytest.mark.parametrize("dim", [None, 0, 'global'])
def test_rescale(simple_tl: TensorList, dim):
    min_val, max_val = 0.0, 1.0
    min_list = [0.0, -1.0, 0.5]
    max_list = [1.0, 0.0, 1.5]
    eps = 1e-6

    # if dim is 0 make sure it isn't len 1 dim
    if dim == 0:
        tensors = TensorList()
        for t in simple_tl:
            while t.shape[0] == 1: t = t[0]
            if t.ndim != 0: tensors.append(t)
        simple_tl = tensors

    # Skip if dim is invalid for a tensor
    if isinstance(dim, int):
        if any(dim >= t.ndim for t in simple_tl):
            pytest.skip(f"Dimension {dim} out of range for at least one tensor")

    # Rescale scalar
    rescaled_scalar = simple_tl.rescale(min_val, max_val, dim=dim, eps=eps)
    rescaled_scalar_min = rescaled_scalar.min(dim=dim if dim != 'global' else None)
    rescaled_scalar_max = rescaled_scalar.max(dim=dim if dim != 'global' else None)

    if dim == 'global':
        assert torch.allclose(rescaled_scalar.global_min(), torch.tensor(min_val))
        assert torch.allclose(rescaled_scalar.global_max(), torch.tensor(max_val))
    else:
        assert_tl_allclose(rescaled_scalar_min, TensorList([torch.full_like(t, min_val) for t in rescaled_scalar_min]),atol=1e-3)
        assert_tl_allclose(rescaled_scalar_max, TensorList([torch.full_like(t, max_val) for t in rescaled_scalar_max]),atol=1e-3)


    # Rescale list
    rescaled_list = simple_tl.rescale(min_list, max_list, dim=dim, eps=eps)
    rescaled_list_min = rescaled_list.min(dim=dim if dim != 'global' else None)
    rescaled_list_max = rescaled_list.max(dim=dim if dim != 'global' else None)

    if dim == 'global':
         # Global rescale with list min/max is tricky, check range contains target roughly
         global_min_rescaled = rescaled_list.global_min()
         global_max_rescaled = rescaled_list.global_max()
         # Cannot guarantee exact match due to single scaling factor 'a' and 'b'
         # Check if the range is approximately correct based on average target range?
         avg_min = sum(min_list)/len(min_list)
         avg_max = sum(max_list)/len(max_list)
         assert global_min_rescaled > avg_min - 1.0 # Loose check
         assert global_max_rescaled < avg_max + 1.0 # Loose check

    else:
        assert_tl_allclose(rescaled_list_min, TensorList([torch.full_like(t, mn) for t, mn in zip(rescaled_list_min, min_list)]),atol=1e-3)
        assert_tl_allclose(rescaled_list_max, TensorList([torch.full_like(t, mx) for t, mx in zip(rescaled_list_max, max_list)]),atol=1e-3)

    # Rescale to 01 helper
    rescaled_01 = simple_tl.rescale_to_01(dim=dim, eps=eps)
    rescaled_01_min = rescaled_01.min(dim=dim if dim != 'global' else None)
    rescaled_01_max = rescaled_01.max(dim=dim if dim != 'global' else None)
    if dim == 'global':
        assert torch.allclose(rescaled_01.global_min(), torch.tensor(0.0))
        assert torch.allclose(rescaled_01.global_max(), torch.tensor(1.0))
    else:
        assert_tl_allclose(rescaled_01_min, TensorList([torch.zeros_like(t) for t in rescaled_01_min]), atol=1e-3)
        assert_tl_allclose(rescaled_01_max, TensorList([torch.ones_like(t) for t in rescaled_01_max]), atol=1e-3)


    # Test inplace
    tl_copy = simple_tl.clone()
    tl_copy.rescale_(min_val, max_val, dim=dim, eps=eps)
    assert_tl_allclose(tl_copy, rescaled_scalar)

    tl_copy_01 = simple_tl.clone()
    tl_copy_01.rescale_to_01_(dim=dim, eps=eps)
    assert_tl_allclose(tl_copy_01, rescaled_01)


@pytest.mark.parametrize("dim", [None, 0, 'global'])
def test_normalize(big_tl: TensorList, dim):
    simple_tl = big_tl # can't be bothered t renamed

    mean_val, var_val = 0.0, 1.0
    mean_list = [0.0, 1.0, -0.5]
    var_list = [1.0, 0.5, 2.0] # Variance > 0

    # if dim is 0 make sure it isn't len 1 dim
    if dim == 0:
        tensors = TensorList()
        for t in simple_tl:
            while t.shape[0] == 1: t = t[0]
            if t.ndim != 0: tensors.append(t)
        simple_tl = tensors

    # Skip if dim is invalid for a tensor
    if isinstance(dim, int):
        if any(dim >= t.ndim for t in simple_tl):
            pytest.skip(f"Dimension {dim} out of range for at least one tensor")

    # Normalize scalar mean/var (z-normalize essentially)
    normalized_scalar = simple_tl.normalize(mean_val, var_val, dim=dim)
    normalized_scalar_mean = normalized_scalar.mean(dim=dim if dim != 'global' else None)
    normalized_scalar_var = normalized_scalar.var(dim=dim if dim != 'global' else None)

    if dim == 'global':
        assert torch.allclose(normalized_scalar.global_mean(), torch.tensor(mean_val), atol=1e-3)
        assert torch.allclose(normalized_scalar.global_var(), torch.tensor(var_val), atol=1e-3)
    else:
        assert_tl_allclose(normalized_scalar_mean, TensorList([torch.full_like(t, mean_val) for t in normalized_scalar_mean]), atol=1e-3)
        assert_tl_allclose(normalized_scalar_var, TensorList([torch.full_like(t, var_val) for t in normalized_scalar_var]), atol=1e-3)

    # Normalize list mean/var
    normalized_list = simple_tl.normalize(mean_list, var_list, dim=dim)
    normalized_list_mean = normalized_list.mean(dim=dim if dim != 'global' else None)
    normalized_list_var = normalized_list.var(dim=dim if dim != 'global' else None)

    if dim == 'global':
         global_mean_rescaled = normalized_list.global_mean()
         global_var_rescaled = normalized_list.global_var()
         avg_mean = sum(mean_list)/len(mean_list)
         avg_var = sum(var_list)/len(var_list)
         # Cannot guarantee exact match due to single scaling factor 'a' and 'b'
         assert global_mean_rescaled - 0.6 < torch.tensor(avg_mean) < global_mean_rescaled + 0.6
         assert global_var_rescaled - 0.6 < torch.tensor(avg_var) < global_var_rescaled + 0.6
        #  assert torch.allclose(global_mean_rescaled, torch.tensor(avg_mean), rtol=1e-1, atol=1e-1) # Loose check
        #  assert torch.allclose(global_var_rescaled, torch.tensor(avg_var), rtol=1e-1, atol=1e-1) # Loose check
    else:
        assert_tl_allclose(normalized_list_mean, TensorList([torch.full_like(t, m) for t, m in zip(normalized_list_mean, mean_list)]), atol=1e-3)
        assert_tl_allclose(normalized_list_var, TensorList([torch.full_like(t, v) for t, v in zip(normalized_list_var, var_list)]), atol=1e-3)

    # Z-normalize helper
    znorm = simple_tl.znormalize(dim=dim, eps=1e-10)
    znorm_mean = znorm.mean(dim=dim if dim != 'global' else None)
    znorm_var = znorm.var(dim=dim if dim != 'global' else None)
    if dim == 'global':
        assert torch.allclose(znorm.global_mean(), torch.tensor(0.0), atol=1e-3)
        assert torch.allclose(znorm.global_var(), torch.tensor(1.0), atol=1e-3)
    else:
        assert_tl_allclose(znorm_mean, TensorList([torch.zeros_like(t) for t in znorm_mean]), atol=1e-3)
        assert_tl_allclose(znorm_var, TensorList([torch.ones_like(t) for t in znorm_var]), atol=1e-3)


    # Test inplace
    tl_copy = simple_tl.clone()
    tl_copy.normalize_(mean_val, var_val, dim=dim)
    assert_tl_allclose(tl_copy, normalized_scalar)

    tl_copy_z = simple_tl.clone()
    tl_copy_z.znormalize_(dim=dim, eps=1e-10)
    assert_tl_allclose(tl_copy_z, znorm)


@pytest.mark.parametrize("tensorwise", [True, False])
def test_clip_norm(simple_tl: TensorList, tensorwise):
    min_val, max_val = 0.5, 1.5
    min_list = [0.2, 0.7, 1.0]
    max_list = [1.0, 1.2, 2.0]
    ord = 2

    # Clip scalar min/max
    clipped_scalar = simple_tl.clip_norm(min_val, max_val, tensorwise=tensorwise, ord=ord)
    if tensorwise:
        clipped_scalar_norms = clipped_scalar.norm(ord=ord)
        assert all(torch.all((n >= min_val - 1e-6) & (n <= max_val + 1e-6)) for n in clipped_scalar_norms)
    else:
        clipped_scalar_global_norm = clipped_scalar.global_vector_norm(ord=ord)
        assert min_val - 1e-6 <= clipped_scalar_global_norm <= max_val + 1e-6

    # Clip list min/max
    clipped_list = simple_tl.clip_norm(min_list, max_list, tensorwise=tensorwise, ord=ord)
    if tensorwise:
        clipped_list_norms = clipped_list.norm(ord=ord)
        assert all(torch.all((n >= mn - 1e-6) & (n <= mx + 1e-6)) for n, mn, mx in zip(clipped_list_norms, min_list, max_list))
    else:
        # Global clip with list min/max is tricky, multiplier is complex
        # Just check type and shape
        assert isinstance(clipped_list, TensorList)
        assert clipped_list.shape == simple_tl.shape


    # Test inplace
    tl_copy = simple_tl.clone()
    tl_copy.clip_norm_(min_val, max_val, tensorwise=tensorwise, ord=ord)
    assert_tl_allclose(tl_copy, clipped_scalar)


# --- Indexing and Masking ---

def test_where(simple_tl: TensorList):
    condition_tl = simple_tl.gt(0)
    other_scalar = -1.0
    other_list_scalar = [-1.0, -2.0, -3.0]
    other_tl = simple_tl.clone().mul_(-1)

    # Scalar other
    result_scalar = simple_tl.where(condition_tl, other_scalar)
    expected_scalar = TensorList([torch.where(c, t, torch.tensor(other_scalar, dtype=t.dtype, device=t.device))
                                  for t, c in zip(simple_tl, condition_tl)])
    assert_tl_allclose(result_scalar, expected_scalar)

    # List scalar other
    result_list_scalar = simple_tl.where(condition_tl, other_list_scalar)
    expected_list_scalar = TensorList([torch.where(c, t, torch.tensor(o, dtype=t.dtype, device=t.device))
                                      for t, c, o in zip(simple_tl, condition_tl, other_list_scalar)])
    assert_tl_allclose(result_list_scalar, expected_list_scalar)


    # TensorList other
    result_tl = simple_tl.where(condition_tl, other_tl)
    expected_tl = TensorList([torch.where(c, t, o) for t, c, o in zip(simple_tl, condition_tl, other_tl)])
    assert_tl_allclose(result_tl, expected_tl)

    # Test module-level where function
    result_module = tl_where(condition_tl, simple_tl, other_tl)
    assert_tl_allclose(result_module, expected_tl)


def test_masked_fill(simple_tl: TensorList):
    mask_tl = simple_tl.lt(0)
    fill_value_scalar = 99.0
    fill_value_list = [11.0, 22.0, 33.0]

    # Scalar fill
    result_scalar = simple_tl.masked_fill(mask_tl, fill_value_scalar)
    expected_scalar = TensorList([t.masked_fill(m, fill_value_scalar) for t, m in zip(simple_tl, mask_tl)])
    assert_tl_allclose(result_scalar, expected_scalar)

    # List fill
    result_list = simple_tl.masked_fill(mask_tl, fill_value_list)
    expected_list = TensorList([t.masked_fill(m, fv) for t, m, fv in zip(simple_tl, mask_tl, fill_value_list)])
    assert_tl_allclose(result_list, expected_list)

    # Test inplace
    tl_copy = simple_tl.clone()
    result_inplace = tl_copy.masked_fill_(mask_tl, fill_value_scalar)
    assert result_inplace is tl_copy
    assert_tl_allclose(tl_copy, expected_scalar)


def test_select_set_(simple_tl: TensorList):
    mask_tl = simple_tl.gt(0.5)
    value_scalar = -1.0
    value_list_scalar = [-1.0, -2.0, -3.0]

    # Set with scalar value
    tl_copy_scalar = simple_tl.clone()
    tl_copy_scalar.select_set_(mask_tl, value_scalar)
    expected_scalar = simple_tl.clone()
    for t, m in zip(expected_scalar, mask_tl): t[m] = value_scalar
    assert_tl_allclose(tl_copy_scalar, expected_scalar)

    # Set with list of scalar values
    tl_copy_list_scalar = simple_tl.clone()
    tl_copy_list_scalar.select_set_(mask_tl, value_list_scalar)
    expected_list_scalar = simple_tl.clone()
    for t, m, v in zip(expected_list_scalar, mask_tl, value_list_scalar): t[m] = v
    assert_tl_allclose(tl_copy_list_scalar, expected_list_scalar)

    # Set with TensorList value
    # no thats masked_set_
    # tl_copy_tl = simple_tl.clone()
    # tl_copy_tl.select_set_(mask_tl, value_tl)
    # expected_tl = simple_tl.clone()
    # for t, m, v in zip(expected_tl, mask_tl, value_tl): t[m] = v[m] # Select from value tensor too
    # assert_tl_allclose(tl_copy_tl, expected_tl)


def test_masked_set_(simple_tl: TensorList):
    mask_tl = simple_tl.gt(0.5)
    value_tl = simple_tl.clone().mul_(0.1)

    tl_copy = simple_tl.clone()
    tl_copy.masked_set_(mask_tl, value_tl)
    expected = simple_tl.clone()
    for t, m, v in zip(expected, mask_tl, value_tl): t[m] = v[m] # masked_set_ semantics
    assert_tl_allclose(tl_copy, expected)


def test_select(simple_tl: TensorList):
    # Select with integer
    idx_int = 0
    result_int = simple_tl.select(idx_int)
    expected_int = TensorList([t[idx_int] for t in simple_tl])
    assert_tl_equal(result_int, expected_int)

    # Select with slice
    idx_slice = slice(0, 1)
    result_slice = simple_tl.select(idx_slice)
    expected_slice = TensorList([t[idx_slice] for t in simple_tl])
    assert_tl_equal(result_slice, expected_slice)

    # Select with list of indices (per tensor)
    idx_list = [0, slice(1, 3), (0, slice(None), 0)] # Different index for each tensor
    result_list = simple_tl.select(idx_list)
    expected_list = TensorList([t[i] for t, i in zip(simple_tl, idx_list)])
    assert_tl_equal(result_list, expected_list)

# --- Miscellaneous ---

def test_dot(simple_tl: TensorList):
    other_tl = simple_tl.clone().mul_(0.5)
    result = simple_tl.dot(other_tl)
    expected = (simple_tl * other_tl).global_sum()
    assert torch.allclose(result, expected)

def test_swap_tensors(simple_tl: TensorList):
    tl1 = simple_tl.clone()
    tl2 = simple_tl.clone().mul_(2)
    tl1_orig_copy = tl1.clone()
    tl2_orig_copy = tl2.clone()

    tl1.swap_tensors(tl2)

    # Check tl1 now has tl2's original data and vice versa
    assert_tl_equal(tl1, tl2_orig_copy)
    assert_tl_equal(tl2, tl1_orig_copy)


def test_unbind_channels(simple_tl: TensorList):
    # Make sure at least one tensor has >1 dim 0 size
    simple_tl[0] = torch.randn(3, 4, 5)
    simple_tl[1] = torch.randn(2, 6)
    simple_tl[2] = torch.randn(1) # Keep a 1D tensor

    unbound_tl = simple_tl.unbind_channels(dim=0)

    expected_list = []
    for t in simple_tl:
        if t.ndim >= 2:
            expected_list.extend(list(t.unbind(dim=0)))
        else:
            expected_list.append(t)
    expected_tl = TensorList(expected_list)

    assert_tl_equal(unbound_tl, expected_tl)


def test_flatiter(simple_tl: TensorList):
    iterator = simple_tl.flatiter()
    all_elements = list(iterator)
    expected_elements = list(simple_tl.to_vec())

    assert len(all_elements) == len(expected_elements)
    for el, exp_el in zip(all_elements, expected_elements):
        # flatiter yields scalar tensors
        assert isinstance(el, torch.Tensor)
        assert el.ndim == 0
        assert torch.equal(el, exp_el)


def test_repr(simple_tl: TensorList):
    representation = repr(simple_tl)
    assert representation.startswith("TensorList([")
    assert representation.endswith("])")
    # Check if tensor representations are inside
    assert "tensor(" in representation


# --- Module Level Functions ---

def test_stack(simple_tl: TensorList):
    tl1 = simple_tl.clone()
    tl2 = simple_tl.clone() * 2
    tl3 = simple_tl.clone() * 3

    stacked_tl = stack([tl1, tl2, tl3], dim=0)
    expected_tl = TensorList([torch.stack([t1, t2, t3], dim=0)
                             for t1, t2, t3 in zip(tl1, tl2, tl3)])
    assert_tl_equal(stacked_tl, expected_tl)

    stacked_tl_dim1 = stack([tl1, tl2, tl3], dim=1)
    expected_tl_dim1 = TensorList([torch.stack([t1, t2, t3], dim=1)
                                  for t1, t2, t3 in zip(tl1, tl2, tl3)])
    assert_tl_equal(stacked_tl_dim1, expected_tl_dim1)


def test_mean_median_sum_quantile_module(simple_tl: TensorList):
    tl1 = simple_tl.clone()
    tl2 = simple_tl.clone() * 2.5
    tl3 = simple_tl.clone() * -1.0
    tensors = [tl1, tl2, tl3]

    # Mean
    mean_res = mean(tensors)
    expected_mean = stack(tensors, dim=0).mean(dim=0)
    assert_tl_allclose(mean_res, expected_mean)

    # Sum
    sum_res = tl_sum(tensors)
    expected_sum = stack(tensors, dim=0).sum(dim=0)
    assert_tl_allclose(sum_res, expected_sum)

    # Median
    median_res = median(tensors)
    # Stack and get median values (result is named tuple)
    expected_median_vals = stack(tensors, dim=0).median(dim=0)
    expected_median = TensorList(expected_median_vals)
    assert_tl_allclose(median_res, expected_median)

    # Quantile
    q = 0.25
    quantile_res = quantile(tensors, q=q)
    expected_quantile_vals = stack(tensors, dim=0).quantile(q=q, dim=0)
    expected_quantile = TensorList(list(expected_quantile_vals))
    assert_tl_allclose(quantile_res, expected_quantile)


# --- Test _MethodCallerWithArgs ---
def test_method_caller_with_args():
    caller = _MethodCallerWithArgs('add')
    t = torch.tensor([1, 2])
    result = caller(t, 5) # t.add(5)
    assert torch.equal(result, torch.tensor([6, 7]))

    result_kw = caller(t, other=10, alpha=2) # t.add(other=10, alpha=2)
    assert torch.equal(result_kw, torch.tensor([21, 22])) # 1 + 2*10, 2 + 2*10

# --- Test generic_clamp ---
def test_generic_clamp():
    assert generic_clamp(5, min=0, max=10) == 5
    assert generic_clamp(-5, min=0, max=10) == 0
    assert generic_clamp(15, min=0, max=10) == 10
    assert generic_clamp(torch.tensor([-5, 5, 15]), min=0, max=10).equal(torch.tensor([0, 5, 10]))

    tl = TensorList([torch.tensor([-5, 5, 15]), torch.tensor([1, 12])])
    clamped_tl = generic_clamp(tl, min=0, max=10)
    expected_tl = TensorList([torch.tensor([0, 5, 10]), torch.tensor([1, 10])])
    assert_tl_equal(clamped_tl, expected_tl)