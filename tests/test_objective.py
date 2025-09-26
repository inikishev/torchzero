import pytest
import torch
from torchzero.core import Objective
from torchzero.utils.tensorlist import TensorList

@torch.no_grad
def test_get_loss():

    # ---------------------------- test that it works ---------------------------- #
    params = [torch.tensor(2.0, requires_grad=True)]
    evaluated = False

    def closure_1(backward=True):
        assert not backward, 'backward = True'

        # ensure closure only evaluates once
        nonlocal evaluated
        assert evaluated is False, 'closure was evaluated twice'
        evaluated = True

        loss = params[0]**2
        if backward:
            params[0].grad = None
            loss.backward()
        else:
            assert not loss.requires_grad, "loss requires grad with backward=False"
        return loss

    obj = Objective(params=params, closure=closure_1, model=None, current_step=0)

    assert obj.loss is None, obj.loss

    assert (loss := obj.get_loss(backward=False)) == 4.0, loss
    assert evaluated, evaluated
    assert loss is obj.loss
    assert obj.loss == 4.0
    assert obj.loss_approx == 4.0
    assert obj.grads is None, obj.grads

    # reevaluate, which should just return already evaluated loss
    assert (loss := obj.get_loss(backward=False)) == 4.0, loss
    assert obj.grads is None, obj.grads


    # ----------------------- test that backward=True works ---------------------- #
    params = [torch.tensor(3.0, requires_grad=True)]
    evaluated = False

    def closure_2(backward=True):
        # ensure closure only evaluates once
        nonlocal evaluated
        assert evaluated is False, 'closure was evaluated twice'
        evaluated = True

        loss = params[0] * 2
        if backward:
            assert loss.requires_grad, "loss does not require grad so `with torch.enable_grad()` context didn't work"
            params[0].grad = None
            loss.backward()
        else:
            assert not loss.requires_grad, "loss requires grad with backward=False"
        return loss

    obj = Objective(params=params, closure=closure_2, model=None, current_step=0)
    assert obj.grads is None, obj.grads
    assert (loss := obj.get_loss(backward=True)) == 6.0, loss
    assert obj.grads is not None
    assert obj.grads[0] == 2.0, obj.grads

    # reevaluate, which should just return already evaluated loss
    assert (loss := obj.get_loss(backward=True)) == 6.0, loss
    assert obj.grads[0] == 2.0, obj.grads

    # get grad, which should just return already evaluated grad
    assert (grad := obj.get_grads())[0] == 2.0, grad
    assert grad is obj.grads, grad

    # get update, which should create and return cloned grad
    assert obj.updates is None
    assert (update := obj.get_updates())[0] == 2.0, update
    assert update is obj.updates
    assert update is not obj.grads
    assert obj.grads is not None
    assert update[0] == obj.grads[0]

@torch.no_grad
def test_get_grad():
    params = [torch.tensor(2.0, requires_grad=True)]
    evaluated = False

    def closure(backward=True):
        # ensure closure only evaluates once
        nonlocal evaluated
        assert evaluated is False, 'closure was evaluated twice'
        evaluated = True

        loss = params[0]**2
        if backward:
            assert loss.requires_grad, "loss does not require grad so `with torch.enable_grad()` context didn't work"
            params[0].grad = None
            loss.backward()
        else:
            assert not loss.requires_grad, "loss requires grad with backward=False"
        return loss

    obj = Objective(params=params, closure=closure, model=None, current_step=0)
    assert (grad := obj.get_grads())[0] == 4.0, grad
    assert grad is obj.grads

    assert obj.loss == 4.0
    assert (loss := obj.get_loss(backward=False)) == 4.0, loss
    assert (loss := obj.get_loss(backward=True)) == 4.0, loss
    assert obj.loss_approx == 4.0

    assert obj.updates is None, obj.updates
    assert (update := obj.get_updates())[0] == 4.0, update

@torch.no_grad
def test_get_update():
    params = [torch.tensor(2.0, requires_grad=True)]
    evaluated = False

    def closure(backward=True):
        # ensure closure only evaluates once
        nonlocal evaluated
        assert evaluated is False, 'closure was evaluated twice'
        evaluated = True

        loss = params[0]**2
        if backward:
            assert loss.requires_grad, "loss does not require grad so `with torch.enable_grad()` context didn't work"
            params[0].grad = None
            loss.backward()
        else:
            assert not loss.requires_grad, "loss requires grad with backward=False"
        return loss

    obj = Objective(params=params, closure=closure, model=None, current_step=0)
    assert obj.updates is None, obj.updates
    assert (update := obj.get_updates())[0] == 4.0, update
    assert update is obj.updates

    assert (grad := obj.get_grads())[0] == 4.0, grad
    assert grad is obj.grads
    assert grad is not update

    assert obj.loss == 4.0
    assert (loss := obj.get_loss(backward=False)) == 4.0, loss
    assert (loss := obj.get_loss(backward=True)) == 4.0, loss
    assert obj.loss_approx == 4.0

    assert (update := obj.get_updates())[0] == 4.0, update


def _assert_objectives_are_same_(o1: Objective, o2: Objective, clone_update: bool):
    for k,v in o1.__dict__.items():
        if not k.startswith('__'):
            # if k == 'post_step_hooks': continue
            if k == 'storage': continue
            elif k == 'updates' and clone_update:
                if o1.updates is None or o2.updates is None:
                    assert o1.updates is None and o2.updates is None, f'`{k}` attribute is not the same, {o1.updates = }, {o2.updates = }'
                else:
                    assert (TensorList(o1.updates) == TensorList(o2.updates)).global_all()
                    assert o1.updates is not o2.updates
            elif k == 'params':
                for p1, p2 in zip(o1.params, o2.params):
                    assert p1.untyped_storage() == p2.untyped_storage()
            else:
                assert getattr(o2, k) is v, f'`{k}` attribute is not the same, {getattr(o1, k) = }, {getattr(o2, k) = }'

def test_var_clone():
    model = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.Linear(2,4))
    def closure(backward): return 1
    obj = Objective(params=list(model.parameters()), closure=closure, model=model, current_step=0)

    _assert_objectives_are_same_(obj, obj.clone(clone_updates=False), clone_update=False)
    _assert_objectives_are_same_(obj, obj.clone(clone_updates=True), clone_update=True)

    obj.grads = TensorList(torch.randn(5))
    _assert_objectives_are_same_(obj, obj.clone(clone_updates=False), clone_update=False)
    _assert_objectives_are_same_(obj, obj.clone(clone_updates=True), clone_update=True)

    obj.updates = TensorList(torch.randn(5) * 2)
    obj.loss = torch.randn(1)
    obj.loss_approx = obj.loss
    _assert_objectives_are_same_(obj, obj.clone(clone_updates=False), clone_update=False)
    _assert_objectives_are_same_(obj, obj.clone(clone_updates=True), clone_update=True)
