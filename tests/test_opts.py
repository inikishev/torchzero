"""snity tests to make sure everything works and converges on basic functions"""
from collections.abc import Callable
from functools import partial

import torch
import torchzero as tz


def _booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def _rosen(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def _ill(x, y):
    return x**2 + y**2 + 1.99999*x*y

def _lstsq(x,y): # specifically for CG and quasi newton methods, staircase effect is more pronounced there
    return (2*x + 3*y - 5)**2 + (5*x - 2*y - 3)**2

funcs = {"booth": (_booth,  (0, -8)), "rosen": (_rosen, (-1.1, 2.5)), "ill": (_ill, (-9, 2.5)), "lstsq": (_lstsq, (-0.9, 0))}
"""{"name": (function, x0)}"""

class _TestModel(torch.nn.Module):
    """sphere with all kinds of parameter shapes, initial loss is 521.2754"""
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        randn = partial(torch.randn, generator=generator)
        params = [
            torch.tensor(1.), torch.tensor([1.]), torch.tensor([[1.]]),
            randn(10), randn(1,10), randn(10,1), randn(1,1,10),randn(1,10,1),randn(1,1,10),
            randn(10,10), randn(4,4,4), randn(3,3,3,3), randn(2,2,2,2,2,2,2),
            randn(10,1,3,1,1),
            torch.zeros(2,2), torch.ones(2,2),
        ]
        self.params = torch.nn.ParameterList(torch.nn.Parameter(t) for t in params)

    def forward(self):
        return torch.sum(torch.stack([p.pow(2).sum() for p in self.params]))

def _run_objective(opt: tz.Modular, objective: Callable, use_closure: bool, steps: int, clear: bool):
    """generic function to run opt on objective and return lowest recorded loss"""
    losses = []
    for i in range(steps):
        if clear and i == steps//2:
            for m in opt.unrolled_modules: m.reset() # clear on middle step to see if there are any issues with it

        if use_closure:
            def closure(backward=True):
                loss = objective()
                if backward:
                    opt.zero_grad()
                    loss.backward()
                return loss
            loss = opt.step(closure)
            assert loss is not None
            assert torch.isfinite(loss), f"Inifinite loss - {[l.item() for l in losses]}"
            losses.append(loss)

        else:
            loss = objective()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), f"Inifinite loss - {[l.item() for l in losses]}"
            losses.append(loss)

    return torch.stack(losses).nan_to_num(0,10000,10000).min()

def _run_func(opt_fn: Callable, func:str, merge: bool, use_closure: bool, steps: int):
    """run optimizer on a test function and return lowest loss"""
    fn, x0 = funcs[func]
    X = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    if merge:
        opt = opt_fn([X])
    else:
        x,y = [i.clone().detach().requires_grad_() for i in X]
        X = (x,y)
        opt = opt_fn(X)

    def objective():
        return fn(*X)

    return _run_objective(opt, objective, use_closure, steps, clear=False)

def _run_sphere(opt_fn: Callable, use_closure:bool, steps:int):
    """run optimizer on sphere test module to test different parameter shapes (common cause of mistakes)"""
    sphere = _TestModel()
    opt = opt_fn(sphere.parameters())
    return _run_objective(opt, sphere, use_closure, steps, clear=True)

def _run(func_opt: Callable, sphere_opt: Callable, needs_closure: bool, func:str, steps: int, loss: float, merge_invariant: bool, sphere_steps: int, sphere_loss: float):
    """Run optimizer on function and sphere test module and check that loss is low enough"""
    tested_sphere = {True: False, False: False} # because shere has no merge
    merged_losses = []
    unmerged_losses = []
    sphere_losses = []

    for merge in [True, False]:
        for use_closure in [True] if needs_closure else [True, False]:
            # print(f"testing with {merge = }, {use_closure = }")
            v = _run_func(func_opt, func, merge, use_closure, steps)
            # print(f'{func} loss after {steps} steps is {v}, target is {loss}')
            assert v <= loss, f"Loss on {func} is {v}, which is above target {loss}. {merge = }, {use_closure = }"
            if merge: merged_losses.append(v)
            else: unmerged_losses.append(v)

            if not tested_sphere[use_closure]:
                tested_sphere[use_closure] = True
                v = _run_sphere(sphere_opt, use_closure, sphere_steps)
                # print(f'sphere loss after {sphere_steps} is {v}, target is {sphere_loss}')
                assert v <= sphere_loss, f"Loss on sphere is {v}, which is above target {sphere_loss}. {merge = }, {use_closure = }"
                sphere_losses.append(v)
            # print()

    # test if losses match
    if merge_invariant: losses = merged_losses + unmerged_losses
    else: losses = merged_losses
    l = losses[0]
    assert all(i == l for i in losses), f"{func} losses don't match: {[l.item() for l in losses]}"

    l = unmerged_losses[0]
    assert all(i == l for i in unmerged_losses), f"Sphere losses don't match: {[l.item() for l in unmerged_losses]}"


    l = sphere_losses[0]
    assert all(i == l for i in sphere_losses), f"Sphere losses don't match: {[l.item() for l in sphere_losses]}"


class Run:
    """
    Holds arguments for a test.

    Args:
        func_opt (Callable): opt for test function e.g. :code:`lambda p: tz.Modular(p, tz.m.Adam())`
        sphere_opt (Callable): opt for sphere e.g. :code:`lambda p: tz.Modular(p, tz.m.Adam(), tz.m.LR(0.1))`
        needs_closure (bool): set to True if opt_fn requires closure
        func (str): what test function to use ("booth", "rosen", "ill")
        steps (int): number of steps to run test function for.
        loss (float): if minimal loss is higher than this then test fails
        merge_invariant (bool): whether the optimizer is invariant to parameters merged or separated.
        sphere_steps (int): how many steps to run sphere for (it has like 1000 params)
        sphere_loss (float): if minimal loss is higher than this then test fails
    """
    def __init__(self, func_opt: Callable, sphere_opt: Callable, needs_closure: bool, func: str, steps: int, loss:float, merge_invariant: bool, sphere_steps:int, sphere_loss:float):
        self.kwargs = locals().copy()
        del self.kwargs['self']
    def test(self): _run(**self.kwargs)

# ----------------------------------- tests ---------------------------------- #


# ------------------------------------ run ----------------------------------- #
def test_opts():
    for v in globals().copy().values():
        if isinstance(v, Run):
            v.test()