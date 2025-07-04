import pytest
import torch
from torchzero.core.module import Var
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

    var = Var(params=params, closure=closure_1, model=None, current_step=0)

    assert var.loss is None, var.loss

    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert evaluated, evaluated
    assert loss is var.loss
    assert var.loss == 4.0
    assert var.loss_approx == 4.0
    assert var.grad is None, var.grad

    # reevaluate, which should just return already evaluated loss
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert var.grad is None, var.grad


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

    var = Var(params=params, closure=closure_2, model=None, current_step=0)
    assert var.grad is None, var.grad
    assert (loss := var.get_loss(backward=True)) == 6.0, loss
    assert var.grad is not None
    assert var.grad[0] == 2.0, var.grad

    # reevaluate, which should just return already evaluated loss
    assert (loss := var.get_loss(backward=True)) == 6.0, loss
    assert var.grad[0] == 2.0, var.grad

    # get grad, which should just return already evaluated grad
    assert (grad := var.get_grad())[0] == 2.0, grad
    assert grad is var.grad, grad

    # get update, which should create and return cloned grad
    assert var.update is None
    assert (update := var.get_update())[0] == 2.0, update
    assert update is var.update
    assert update is not var.grad
    assert var.grad is not None
    assert update[0] == var.grad[0]

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

    var = Var(params=params, closure=closure, model=None, current_step=0)
    assert (grad := var.get_grad())[0] == 4.0, grad
    assert grad is var.grad

    assert var.loss == 4.0
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert (loss := var.get_loss(backward=True)) == 4.0, loss
    assert var.loss_approx == 4.0

    assert var.update is None, var.update
    assert (update := var.get_update())[0] == 4.0, update

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

    var = Var(params=params, closure=closure, model=None, current_step=0)
    assert var.update is None, var.update
    assert (update := var.get_update())[0] == 4.0, update
    assert update is var.update

    assert (grad := var.get_grad())[0] == 4.0, grad
    assert grad is var.grad
    assert grad is not update

    assert var.loss == 4.0
    assert (loss := var.get_loss(backward=False)) == 4.0, loss
    assert (loss := var.get_loss(backward=True)) == 4.0, loss
    assert var.loss_approx == 4.0

    assert (update := var.get_update())[0] == 4.0, update


def _assert_var_are_same_(v1: Var, v2: Var, clone_update: bool):
    for k,v in v1.__dict__.items():
        if not k.startswith('__'):
            # if k == 'post_step_hooks': continue
            if k == 'storage': continue
            if k == 'update' and clone_update:
                if v1.update is None or v2.update is None:
                    assert v1.update is None and v2.update is None, f'{k} is not the same, {v1 = }, {v2 = }'
                else:
                    assert (TensorList(v1.update) == TensorList(v2.update)).global_all()
                    assert v1.update is not v2.update
            else:
                assert getattr(v2, k) is v, f'{k} is not the same, {v1 = }, {v2 = }'

def test_var_clone():
    model = torch.nn.Sequential(torch.nn.Linear(2,2), torch.nn.Linear(2,4))
    def closure(backward): return 1
    var = Var(params=list(model.parameters()), closure=closure, model=model, current_step=0)

    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)

    var.grad = TensorList(torch.randn(5))
    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)

    var.update = TensorList(torch.randn(5) * 2)
    var.loss = torch.randn(1)
    var.loss_approx = var.loss
    _assert_var_are_same_(var, var.clone(clone_update=False), clone_update=False)
    _assert_var_are_same_(var, var.clone(clone_update=True), clone_update=True)
