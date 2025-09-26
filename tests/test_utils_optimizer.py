from collections.abc import Iterable
from typing import Any
from functools import partial
import pytest
import torch
from torchzero.utils.optimizer import (
    get_group_vals,
    get_params,
    get_state_vals,
)


def _assert_same_storage_(seq1: Iterable[torch.Tensor], seq2: Iterable[torch.Tensor]):
    seq1=tuple(seq1)
    seq2=tuple(seq2)
    assert len(seq1) == len(seq2), f'lengths do not match: {len(seq1)} != {len(seq2)}'
    for t1, t2 in zip(seq1, seq2):
        assert t1 is t2

def _assert_equals_different_storage_(seq1: Iterable[torch.Tensor], seq2: Iterable[torch.Tensor]):
    seq1=tuple(seq1)
    seq2=tuple(seq2)
    assert len(seq1) == len(seq2), f'lengths do not match: {len(seq1)} != {len(seq2)}'
    for t1, t2 in zip(seq1, seq2):
        assert t1 is not t2
        assert (t1 == t2).all()

def test_assert_compare_tensors():
    t1 = [torch.randn(1, 3) for _ in range(10)]
    t2 = [torch.randn(1, 3) for _ in range(10)]

    _assert_same_storage_(t1, t1)
    _assert_same_storage_(t2, t2)

    with pytest.raises(AssertionError):
        _assert_same_storage_(t1, t2)


def test_get_params():
    param_groups = [
        {'params': [torch.randn(1, 1, requires_grad=True), torch.randn(1, 2, requires_grad=True)]},
        {'params': [torch.randn(2, 1, requires_grad=True), torch.randn(2, 2, requires_grad=False)], "lr": 0.1},
        {'params': [torch.randn(3, 1, requires_grad=False)], 'lr': 0.001, 'betas': (0.9, 0.99)},
    ]
    param_groups[0]['params'][0].grad = torch.randn(1, 1)

    params = get_params(param_groups, mode = 'requires_grad', cls = list)
    _assert_same_storage_(params, [*param_groups[0]['params'], param_groups[1]['params'][0]])

    params = get_params(param_groups, mode = 'has_grad', cls = list)
    _assert_same_storage_(params, [param_groups[0]['params'][0]])

    params = get_params(param_groups, mode = 'all', cls = list)
    _assert_same_storage_(params, [*param_groups[0]['params'], *param_groups[1]['params'], *param_groups[2]['params']])

def test_get_group_vals():
    param_groups = [
        {'params': [torch.randn(2, 1, requires_grad=True), torch.randn(2, 2, requires_grad=True)], "lr": 0.1, 'beta': 0.95, 'eps': 1e-8},
        {'params': [torch.randn(1, 1, requires_grad=True), torch.randn(1, 2, requires_grad=False)], 'lr': 0.01, 'beta': 0.99, 'eps': 1e-7},
        {'params': [torch.randn(3, 1, requires_grad=False)], 'lr': 0.001, 'beta': 0.999, 'eps': 1e-6},
    ]
    param_groups[0]['params'][0].grad = torch.randn(2, 1)


    lr = get_group_vals(param_groups, 'lr', mode = 'requires_grad', cls = list)
    assert lr == [0.1, 0.1, 0.01], lr

    lr, beta = get_group_vals(param_groups, 'lr', 'beta', mode = 'requires_grad', cls = list)
    assert lr == [0.1, 0.1, 0.01], lr
    assert beta == [0.95, 0.95, 0.99], beta

    lr, beta, eps = get_group_vals(param_groups, ('lr', 'beta', 'eps'), mode = 'requires_grad', cls = list)
    assert lr == [0.1, 0.1, 0.01], lr
    assert beta == [0.95, 0.95, 0.99], beta
    assert eps == [1e-8, 1e-8, 1e-7], eps

    lr = get_group_vals(param_groups, 'lr', mode = 'has_grad', cls = list)
    assert lr == [0.1], lr

    lr, beta, eps = get_group_vals(param_groups, 'lr', 'beta', 'eps', mode = 'all', cls = list)
    assert lr == [0.1, 0.1, 0.01, 0.01, 0.001], lr
    assert beta == [0.95, 0.95, 0.99, 0.99, 0.999], beta
    assert eps == [1e-8, 1e-8, 1e-7, 1e-7, 1e-6], eps

def test_get_state_vals():
    # accessing state values of a single parameter, which acts as the key, so no tensors are passed
    tensor = torch.randn(3,3)
    state = {tensor: {'exp_avg': torch.ones_like(tensor)}}
    existing_cov_exp_avg = state[tensor]['exp_avg']
    cov_exp_avg, cov_exp_avg_sq = get_state_vals(state, [tensor], ('exp_avg', 'exp_avg_sq'), init = [torch.zeros_like, lambda x: torch.full_like(x, 2)])
    assert torch.allclose(cov_exp_avg[0], torch.ones_like(tensor))
    assert torch.allclose(cov_exp_avg_sq[0], torch.full_like(tensor, 2))
    assert cov_exp_avg[0] is existing_cov_exp_avg
    assert state[tensor]['exp_avg'] is existing_cov_exp_avg
    assert state[tensor]['exp_avg_sq'] is cov_exp_avg_sq[0]

    # accessing state values of multiple parameters
    parameters = [torch.randn(i,2) for i in range(1, 11)]
    state = {p: {} for p in parameters}
    exp_avgs = get_state_vals(state, parameters, 'exp_avg', cls=list)
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert len(exp_avgs) == 10, len(exp_avgs)
    assert all(torch.allclose(a, torch.zeros_like(parameters[i])) for i, a in enumerate(exp_avgs))
    exp_avgs2 = get_state_vals(state, parameters, 'exp_avg', cls=list)
    _assert_same_storage_(exp_avgs, exp_avgs2)

    # per-parameter inits
    parameters = [torch.full((i,2), fill_value=i**2) for i in range(1, 11)]
    state = {p: {} for p in parameters}
    exp_avgs = get_state_vals(state, parameters, 'exp_avg', init = [partial(torch.full_like, fill_value=i) for i in range(10)], cls=list)
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert len(exp_avgs) == 10, len(exp_avgs)
    assert all(torch.allclose(a, torch.full_like(parameters[i], i)) for i, a in enumerate(exp_avgs)), exp_avgs
    exp_avgs2 = get_state_vals(state, parameters, 'exp_avg', cls=list)
    _assert_same_storage_(exp_avgs, exp_avgs2)

    # per-parmeter init with a list
    parameters = [torch.full((i,2), fill_value=i**2) for i in range(1, 11)]
    state = {p: {} for p in parameters}
    inits = [torch.full([i], fill_value=i) for i in range(1, 11)]
    exp_avgs = get_state_vals(state, parameters, 'exp_avg', init = inits, cls=list)
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert len(exp_avgs) == 10, len(exp_avgs)
    _assert_equals_different_storage_(inits, exp_avgs) # inits are cloned
    exp_avgs2 = get_state_vals(state, parameters, 'exp_avg', cls=list)
    _assert_same_storage_(exp_avgs, exp_avgs2)

    # init with a value
    parameters = [torch.full((i,2), fill_value=i**2) for i in range(1, 11)]
    state = {p: {} for p in parameters}
    inits = 1
    exp_avgs = get_state_vals(state, parameters, 'exp_avg', init = inits, cls=list)
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert len(exp_avgs) == 10, len(exp_avgs)
    assert all(v==1 for v in exp_avgs), exp_avgs
    assert exp_avgs == get_state_vals(state, parameters, 'exp_avg', cls=list) # no init because already initialized

    # accessing multiple keys
    parameters = [torch.randn(i,2) for i in range(1,11)]
    state = {p: {} for p in parameters}
    exp_avgs, exp_avg_sqs, max_avgs = get_state_vals(state, parameters, 'exp_avg', 'exp_avg_sq', 'max_avg', cls=list)
    assert len(exp_avgs) == len(exp_avg_sqs) == len(max_avgs) == 10
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert isinstance(exp_avg_sqs, list), type(exp_avg_sqs)
    assert isinstance(max_avgs, list), type(max_avgs)
    assert all(torch.allclose(a, torch.zeros_like(parameters[i])) for i, a in enumerate(exp_avgs))
    assert all(torch.allclose(a, torch.zeros_like(parameters[i])) for i, a in enumerate(exp_avg_sqs))
    assert all(torch.allclose(a, torch.zeros_like(parameters[i])) for i, a in enumerate(max_avgs))
    exp_avgs2 = get_state_vals(state, parameters, 'exp_avg', cls=list)
    exp_avg_sqs2 = get_state_vals(state, parameters, 'exp_avg_sq', cls=list)
    max_avgs2 = get_state_vals(state, parameters, 'max_avg', cls=list)
    _assert_same_storage_(exp_avgs, exp_avgs2)
    _assert_same_storage_(exp_avg_sqs, exp_avg_sqs2)
    _assert_same_storage_(max_avgs, max_avgs2)

    # per-key init
    parameters = [torch.randn(i,2) for i in range(1,11)]
    state = {p: {} for p in parameters}
    exp_avgs, exp_avg_sqs, max_avgs = get_state_vals(state, parameters, 'exp_avg', 'exp_avg_sq', 'max_avg', init=(4,5,5.5), cls=list)
    assert len(exp_avgs) == len(exp_avg_sqs) == len(max_avgs) == 10
    assert isinstance(exp_avgs, list), type(exp_avgs)
    assert isinstance(exp_avg_sqs, list), type(exp_avg_sqs)
    assert isinstance(max_avgs, list), type(max_avgs)
    assert all(v==4 for v in exp_avgs), exp_avgs
    assert all(v==5 for v in exp_avg_sqs), exp_avg_sqs
    assert all(v==5.5 for v in max_avgs), max_avgs
    assert exp_avgs == get_state_vals(state, parameters, 'exp_avg', cls=list)
    assert exp_avg_sqs == get_state_vals(state, parameters, 'exp_avg_sq', cls=list)
    assert max_avgs == get_state_vals(state, parameters, 'max_avg', cls=list)