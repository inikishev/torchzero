import pytest
import torch
from torchzero.core import Objective
from torchzero.utils.tensorlist import TensorList

@torch.no_grad
def test_var_get_loss():

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

    var = Objective(params=params, closure=closure_1, model=None, current_step=0)

    assert var.loss is None, var.loss

    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert evaluated, evaluated
    assert loss is var.loss
    assert var.loss == 4.0
    assert var.loss_approx == 4.0
    assert var.grads is None, var.grads

    # reevaluate, which should just return already evaluated loss
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert var.grads is None, var.grads


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

    var = Objective(params=params, closure=closure_2, model=None, current_step=0)
    assert var.grads is None, var.grads
    assert (loss := var.get_loss(backward=True)) == 6.0, loss
    assert var.grads is not None
    assert var.grads[0] == 2.0, var.grads

    # reevaluate, which should just return already evaluated loss
    assert (loss := var.get_loss(backward=True)) == 6.0, loss
    assert var.grads[0] == 2.0, var.grads

    # get grad, which should just return already evaluated grad
    assert (grad := var.get_grads())[0] == 2.0, grad
    assert grad is var.grads, grad

    # get update, which should create and return cloned grad
    assert var.updates is None
    assert (update := var.get_updates())[0] == 2.0, update
    assert update is var.updates
    assert update is not var.grads
    assert var.grads is not None
    assert update[0] == var.grads[0]

@torch.no_grad
def test_var_get_grad():
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

    var = Objective(params=params, closure=closure, model=None, current_step=0)
    assert (grad := var.get_grads())[0] == 4.0, grad
    assert grad is var.grads

    assert var.loss == 4.0
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert (loss := var.get_loss(backward=True)) == 4.0, loss
    assert var.loss_approx == 4.0

    assert var.updates is None, var.updates
    assert (update := var.get_updates())[0] == 4.0, update

@torch.no_grad
def test_var_get_update():
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

    var = Objective(params=params, closure=closure, model=None, current_step=0)
    assert var.updates is None, var.updates
    assert (update := var.get_updates())[0] == 4.0, update
    assert update is var.updates

    assert (grad := var.get_grads())[0] == 4.0, grad
    assert grad is var.grads
    assert grad is not update

    assert var.loss == 4.0
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert (loss := var.get_loss(backward=True)) == 4.0, loss
    assert var.loss_approx == 4.0

    assert (update := var.get_updates())[0] == 4.0, update


def _assert_var_are_same_(v1: Objective, v2: Objective, clone_update: bool):
    for k,v in v1.__dict__.items():
        if not k.startswith('__'):
            # if k == 'post_step_hooks': continue
            if k == 'storage': continue
            if k == 'update' and clone_update:
                if v1.updates is None or v2.updates is None:
                    assert v1.updates is None and v2.updates is None, f'{k} is not the same, {v1 = }, {v2 = }'
                else:
                    assert (TensorList(v1.updates) == TensorList(v2.updates)).global_all()
                    assert v1.updates is not v2.updates
            else:
                assert getattr(v2, k) is v, f'{k} is not the same, {v1 = }, {v2 = }'

def test_var_clone():
    model = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.Linear(2,4))
    def closure(backward): return 1
    var = Objective(params=list(model.parameters()), closure=closure, model=model, current_step=0)

    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)

    var.grads = TensorList(torch.randn(5))
    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)

    var.updates = TensorList(torch.randn(5) * 2)
    var.loss = torch.randn(1)
    var.loss_approx = var.loss
    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)
