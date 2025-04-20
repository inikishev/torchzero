from collections.abc import Iterable

import torch
from torchzero.core.module import Module, _make_param_groups
from torchzero.utils.optimizer import get_params
from torchzero.utils.params import _add_defaults_to_param_groups_

def _assert_same_storage_(seq1: Iterable[torch.Tensor], seq2: Iterable[torch.Tensor]):
    seq1=tuple(seq1)
    seq2=tuple(seq2)
    assert len(seq1) == len(seq2), f'lengths do not match: {len(seq1)} != {len(seq2)}'
    for t1, t2 in zip(seq1, seq2):
        assert t1 is t2


def test_process_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Linear(6, 3))

    # iterable of parameters
    _assert_same_storage_(model.parameters(), get_params(_make_param_groups(model.parameters(), differentiable=False), 'all'))

    # named parameters
    _assert_same_storage_(model.parameters(), get_params(_make_param_groups(model.named_parameters(), differentiable=False), 'all'))

    # param groups
    param_groups = [{'params': model[0].parameters(), 'lr': 0.1}, {'params': model[1].parameters()}]
    _assert_same_storage_(model.parameters(), get_params(_make_param_groups(param_groups, differentiable=False), 'all'))

    # check that param groups dict is correct
    param_groups = [
        {'params': model[0].parameters(), 'lr': 0.1},
        {'params': model[1].parameters()}
    ]
    expected = [
        {'params': list(model[0].parameters()), 'lr': 0.1},
        {'params': list(model[1].parameters())}
    ]
    assert _make_param_groups(param_groups, differentiable=False) == expected

    # named params
    _names = {'param_names': ['weight','bias']}
    param_groups = [
        {'params': model[0].named_parameters(), 'lr': 0.1},
        {'params': model[1].named_parameters()}
    ]
    expected = [
        {'params': list(model[0].parameters()), 'lr': 0.1, **_names},
        {'params': list(model[1].parameters()), 'lr': 0.01, **_names}
    ]
    assert _add_defaults_to_param_groups_(_make_param_groups(param_groups, differentiable=False), {"lr": 0.01}) == expected
