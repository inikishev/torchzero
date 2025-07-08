"""
Sanity tests to make sure everything works.

This will show major convergence regressions, but that is not the main purpose. Mainly this makes sure modules
don't error or become unhinged with different parameter shapes.
"""
from collections.abc import Callable
from functools import partial

import pytest
import torch
import torchzero as tz

PRINT = False # set to true in nbs

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
            assert torch.isfinite(loss), f"{opt}: Inifinite loss - {[l.item() for l in losses]}"
            losses.append(loss)

        else:
            loss = objective()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), f"{opt}: Inifinite loss - {[l.item() for l in losses]}"
            losses.append(loss)

    losses.append(objective())
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

    return _run_objective(opt, objective, use_closure, steps, clear=False), opt

def _run_sphere(opt_fn: Callable, use_closure:bool, steps:int):
    """run optimizer on sphere test module to test different parameter shapes (common cause of mistakes)"""
    sphere = _TestModel()
    opt = opt_fn(sphere.parameters())
    return _run_objective(opt, sphere, use_closure, steps, clear=True), opt

def _run(func_opt: Callable, sphere_opt: Callable, needs_closure: bool, func:str, steps: int, loss: float, merge_invariant: bool, sphere_steps: int, sphere_loss: float):
    """Run optimizer on function and sphere test module and check that loss is low enough"""
    tested_sphere = {True: False, False: False} # because shere has no merge
    merged_losses = []
    unmerged_losses = []
    sphere_losses = []

    for merge in [True, False]:
        for use_closure in [True] if needs_closure else [True, False]:
            if PRINT: print(f"testing with {merge = }, {use_closure = }")
            v,opt = _run_func(func_opt, func, merge, use_closure, steps)
            if PRINT: print(f'{func} loss after {steps} steps is {v}, target is {loss}')
            assert v <= loss, f"{opt}: Loss on {func} is {v}, which is above target {loss}. {merge = }, {use_closure = }"
            if merge: merged_losses.append(v)
            else: unmerged_losses.append(v)

            if not tested_sphere[use_closure]:
                tested_sphere[use_closure] = True
                v,opt = _run_sphere(sphere_opt, use_closure, sphere_steps)
                if PRINT: print(f'sphere loss after {sphere_steps} is {v}, target is {sphere_loss}')
                assert v <= sphere_loss, f"{opt}: Loss on sphere is {v}, which is above target {sphere_loss}. {merge = }, {use_closure = }"
                sphere_losses.append(v)
            if PRINT: print()

    # test if losses match
    if merge_invariant: losses = merged_losses + unmerged_losses
    else: losses = merged_losses
    l = losses[0]
    assert all(i == l for i in losses), f"{func} losses don't match: {[l.item() for l in losses]}"

    l = unmerged_losses[0]
    assert all(i == l for i in unmerged_losses), f"Sphere losses don't match: {[l.item() for l in unmerged_losses]}"


    l = sphere_losses[0]
    assert all(i == l for i in sphere_losses), f"Sphere losses don't match: {[l.item() for l in sphere_losses]}"

RUNS = []
"""Whenever a Run is created (__init__ is called) it gets appened to this"""

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
        RUNS.append(self)
    def test(self): _run(**self.kwargs)

# target losses for all of those are set to just above what they reach
# ---------------------------------------------------------------------------- #
#                                     tests                                    #
# ---------------------------------------------------------------------------- #
# ----------------------------- clipping/clipping ---------------------------- #
ClipValue = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipValue(1), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipValue(1), tz.m.LR(1)),
    needs_closure=False,
    func='booth', steps=50, loss=0, merge_invariant=True,
    sphere_steps=10, sphere_loss=50,
)
ClipNorm = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNorm(1), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNorm(1), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=2, merge_invariant=False,
    sphere_steps=10, sphere_loss=0,
)
ClipNorm_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNorm(1, dim='global'), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNorm(1, dim='global'), tz.m.LR(3)),
    needs_closure=False,
    func='booth', steps=50, loss=2, merge_invariant=True,
    sphere_steps=10, sphere_loss=2,
)
Normalize = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Normalize(1), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Normalize(1), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=2, merge_invariant=False,
    sphere_steps=10, sphere_loss=15,
)
Normalize_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Normalize(1, dim='global'), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Normalize(1, dim='global'), tz.m.LR(4)),
    needs_closure=False,
    func='booth', steps=50, loss=2, merge_invariant=True,
    sphere_steps=10, sphere_loss=2,
)
Centralize = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Centralize(min_size=3), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Centralize(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=False,
    sphere_steps=10, sphere_loss=10,
)
Centralize_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Centralize(min_size=3, dim='global'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Centralize(dim='global'), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=1, loss=1000, merge_invariant=True,
    sphere_steps=10, sphere_loss=10,
)

# --------------------------- clipping/ema_clipping -------------------------- #
ClipNormByEMA = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNormByEMA(), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNormByEMA(), tz.m.LR(5)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-5, merge_invariant=False,
    sphere_steps=10, sphere_loss=0.1,
)
ClipNormByEMA_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNormByEMA(tensorwise=False), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNormByEMA(tensorwise=False), tz.m.LR(5)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-5, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)
NormalizeByEMA = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.NormalizeByEMA(), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.NormalizeByEMA(), tz.m.LR(5)),
    needs_closure=False,
    func='booth', steps=50, loss=1, merge_invariant=False,
    sphere_steps=10, sphere_loss=0.1,
)
NormalizeByEMA_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.NormalizeByEMA(tensorwise=False), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.NormalizeByEMA(tensorwise=False), tz.m.LR(5)),
    needs_closure=False,
    func='booth', steps=50, loss=1, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)
ClipValueByEMA = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipValueByEMA(), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipValueByEMA(), tz.m.LR(4)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-5, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.03,
)
# ------------------------- clipping/growth_clipping ------------------------- #
ClipValueGrowth = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipValueGrowth(), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipValueGrowth(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=100,
)
ClipValueGrowth_additive = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipValueGrowth(add=1, mul=None), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipValueGrowth(add=1, mul=None), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=10,
)
ClipNormGrowth = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=False,
    sphere_steps=10, sphere_loss=10,
)
ClipNormGrowth_additive = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(add=1,mul=None), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(add=1,mul=None), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=False,
    sphere_steps=10, sphere_loss=10,
)
ClipNormGrowth_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(parameterwise=False), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ClipNormGrowth(parameterwise=False), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=10,
)

# -------------------------- grad_approximation/fdm -------------------------- #
FDM_central2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='central2'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)
FDM_forward2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='forward2'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='forward2'), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)
FDM_backward2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='backward2'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='backward2'), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)
FDM_forward3 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='forward3'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='forward3'), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)
FDM_backward3 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='backward3'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='backward3'), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)
FDM_central4 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='central4'), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.FDM(formula='central4'), tz.m.LR(0.1)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=2, sphere_loss=340,
)

# -------------------------- grad_approximation/rfdm ------------------------- #
RandomizedFDM_central2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
RandomizedFDM_forward2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='forward2', seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='forward2', seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
RandomizedFDM_backward2 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='backward2', seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='backward2', seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
RandomizedFDM_forward3 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='forward3', seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='forward3', seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
RandomizedFDM_backward3 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='backward3', seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='backward3', seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
RandomizedFDM_central4 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='central4', seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(formula='central4', seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=10, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)

RandomizedFDM_4samples = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, seed=0), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-5, merge_invariant=True,
    sphere_steps=100, sphere_loss=400,
)
RandomizedFDM_4samples_lerp = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, beta=0.99, seed=0), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, beta=0.9, seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-5, merge_invariant=True,
    sphere_steps=100, sphere_loss=505,
)
RandomizedFDM_4samples_no_pre_generate = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, pre_generate=False, seed=0), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomizedFDM(n_samples=4, pre_generate=False, seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-5, merge_invariant=True,
    sphere_steps=100, sphere_loss=400,
)
MeZO = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MeZO(), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MeZO(), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=5, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
MeZO_4samples = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MeZO(n_samples=4), tz.m.LR(0.02)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MeZO(n_samples=4), tz.m.LR(0.005)),
    needs_closure=True,
    func='booth', steps=50, loss=1, merge_invariant=True,
    sphere_steps=100, sphere_loss=250,
)
# -------------------- grad_approximation/forward_gradient ------------------- #
ForwardGradient = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=40, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
ForwardGradient_forward = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0, jvp_method='forward'), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0, jvp_method='forward'), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=40, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
ForwardGradient_central = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0, jvp_method='central'), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(seed=0, jvp_method='central'), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=40, merge_invariant=True,
    sphere_steps=100, sphere_loss=450,
)
ForwardGradient_4samples = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(n_samples=4, seed=0), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(n_samples=4, seed=0), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=0.1, merge_invariant=True,
    sphere_steps=100, sphere_loss=400,
)
ForwardGradient_4samples_no_pre_generate = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(n_samples=4, seed=0, pre_generate=False), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ForwardGradient(n_samples=4, seed=0, pre_generate=False), tz.m.LR(0.001)),
    needs_closure=True,
    func='booth', steps=50, loss=0.1, merge_invariant=True,
    sphere_steps=100, sphere_loss=400,
)

# ------------------------- line_search/backtracking ------------------------- #
Backtracking = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Backtracking()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Backtracking()),
    needs_closure=True,
    func='booth', steps=50, loss=0, merge_invariant=True,
    sphere_steps=2, sphere_loss=0,
)
Backtracking_try_negative = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Mul(-1), tz.m.Backtracking(try_negative=True)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Mul(-1), tz.m.Backtracking(try_negative=True)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-9, merge_invariant=True,
    sphere_steps=2, sphere_loss=1e-10,
)
AdaptiveBacktracking = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.AdaptiveBacktracking()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.AdaptiveBacktracking()),
    needs_closure=True,
    func='booth', steps=50, loss=1e-12, merge_invariant=True,
    sphere_steps=2, sphere_loss=1e-10,
)
AdaptiveBacktracking_try_negative = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Mul(-1), tz.m.AdaptiveBacktracking(try_negative=True)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Mul(-1), tz.m.AdaptiveBacktracking(try_negative=True)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-8, merge_invariant=True,
    sphere_steps=2, sphere_loss=1e-10,
)
# ----------------------------- line_search/scipy ---------------------------- #
ScipyMinimizeScalar = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ScipyMinimizeScalar(maxiter=10)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.AdaptiveBacktracking(maxiter=10)),
    needs_closure=True,
    func='booth', steps=50, loss=1e-2, merge_invariant=True,
    sphere_steps=2, sphere_loss=0,
)

# ------------------------- line_search/strong_wolfe ------------------------- #
StrongWolfe = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.StrongWolfe()),
    needs_closure=True,
    func='booth', steps=50, loss=0, merge_invariant=True,
    sphere_steps=2, sphere_loss=0,
)

# ----------------------------------- lr/lr ---------------------------------- #
LR = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
StepSize = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.StepSize(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.StepSize(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
Warmup = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Warmup(steps=50, end_lr=0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Warmup(steps=10)),
    needs_closure=False,
    func='booth', steps=50, loss=0.003, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.05,
)
# ------------------------------- lr/step_size ------------------------------- #
PolyakStepSize = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.PolyakStepSize()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.PolyakStepSize()),
    needs_closure=True,
    func='booth', steps=50, loss=1e-7, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.002,
)
RandomStepSize = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomStepSize(0,0.1, seed=0)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomStepSize(0,0.1, seed=0)),
    needs_closure=False,
    func='booth', steps=50, loss=0.0005, merge_invariant=True,
    sphere_steps=10, sphere_loss=100,
)
RandomStepSize_parameterwise = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.RandomStepSize(0,0.1, parameterwise=True, seed=0)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.RandomStepSize(0,0.1, parameterwise=True, seed=0)),
    needs_closure=False,
    func='booth', steps=50, loss=0.0005, merge_invariant=False,
    sphere_steps=10, sphere_loss=100,
)

# ---------------------------- momentum/averaging ---------------------------- #
Averaging = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Averaging(10), tz.m.LR(0.02)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Averaging(10), tz.m.LR(0.2)),
    needs_closure=False,
    func='booth', steps=50, loss=0.5, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.05,
)
WeightedAveraging = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.WeightedAveraging([1,0.75,0.5,0.25,0]), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.WeightedAveraging([1,0.75,0.5,0.25,0]), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=1, merge_invariant=True,
    sphere_steps=10, sphere_loss=2,
)
MedianAveraging = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MedianAveraging(10), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MedianAveraging(10), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=0.005, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

# ----------------------------- momentum/cautious ---------------------------- #
Cautious = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.Cautious(), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.Cautious(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0.003, merge_invariant=True,
    sphere_steps=10, sphere_loss=2,
)
UpdateGradientSignConsistency = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.Mul(tz.m.UpdateGradientSignConsistency()), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.Mul(tz.m.UpdateGradientSignConsistency()), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0.003, merge_invariant=True,
    sphere_steps=10, sphere_loss=2,
)
IntermoduleCautious = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.IntermoduleCautious(tz.m.NAG(), tz.m.BFGS(ptol_reset=True)), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.IntermoduleCautious(tz.m.NAG(), tz.m.BFGS(ptol_reset=True)), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-4, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)
ScaleByGradCosineSimilarity = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.ScaleByGradCosineSimilarity(), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(0.9), tz.m.ScaleByGradCosineSimilarity(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0.1, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)
ScaleModulesByCosineSimilarity = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.ScaleModulesByCosineSimilarity(tz.m.HeavyBall(0.9), tz.m.BFGS(ptol_reset=True)),tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.ScaleModulesByCosineSimilarity(tz.m.HeavyBall(0.9), tz.m.BFGS(ptol_reset=True)),tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0.005, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)

# ------------------------- momentum/matrix_momentum ------------------------- #
MatrixMomentum_forward = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='forward'), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='forward'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.05, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
MatrixMomentum_forward = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='central'), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='central'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.05, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
MatrixMomentum_forward = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='autograd'), tz.m.LR(0.01)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.MatrixMomentum(hvp_method='autograd'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.05, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

AdaptiveMatrixMomentum_forward = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='forward'), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='forward'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.002, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
AdaptiveMatrixMomentum_central = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='central'), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='central'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.002, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
AdaptiveMatrixMomentum_autograd = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='autograd'), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.AdaptiveMatrixMomentum(hvp_method='autograd'), tz.m.LR(0.5)),
    needs_closure=True,
    func='booth', steps=50, loss=0.002, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

# EMA, momentum are covered by test_identical
# --------------------------------- ops/misc --------------------------------- #
Previous = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Previous(10), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Previous(3), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=15, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
GradSign = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.GradSign(), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.GradSign(), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=0.0002, merge_invariant=True,
    sphere_steps=10, sphere_loss=0.1,
)
UpdateSign = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.UpdateSign(), tz.m.LR(0.05)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.UpdateSign(), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=0.01, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
GradAccumulation = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.GradientAccumulation(tz.m.LR(0.05), 10), ),
    sphere_opt=lambda p: tz.Modular(p, tz.m.GradientAccumulation(tz.m.LR(0.5), 10), ),
    needs_closure=False,
    func='booth', steps=50, loss=25, merge_invariant=True,
    sphere_steps=20, sphere_loss=1e-11,
)
NegateOnLossIncrease = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.LR(0.02), tz.m.NegateOnLossIncrease(True),),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HeavyBall(), tz.m.LR(0.1), tz.m.NegateOnLossIncrease(True),),
    needs_closure=True,
    func='booth', steps=50, loss=0.1, merge_invariant=True,
    sphere_steps=20, sphere_loss=0.001,
)
# -------------------------------- misc/switch ------------------------------- #
Alternate = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Alternate(tz.m.Adagrad(), tz.m.Adam(), tz.m.RMSprop()), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Alternate(tz.m.Adagrad(), tz.m.Adam(), tz.m.RMSprop()), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=1, merge_invariant=True,
    sphere_steps=20, sphere_loss=20,
)

# ------------------------------ optimizers/adam ----------------------------- #
Adam = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Adam(), tz.m.LR(0.5)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Adam(), tz.m.LR(0.2)),
    needs_closure=False,
    func='rosen', steps=50, loss=4, merge_invariant=True,
    sphere_steps=20, sphere_loss=4,
)
# ------------------------------ optimizers/soap ----------------------------- #
SOAP = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.SOAP(), tz.m.LR(0.4)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.SOAP(), tz.m.LR(1)),
    needs_closure=False,
    func='rosen', steps=50, loss=4, merge_invariant=False,
    sphere_steps=20, sphere_loss=25, # merge and unmerge lrs are very different so need to test convergence separately somewhere
)
# ------------------------------ optimizers/lion ----------------------------- #
Lion = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Lion(), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Lion(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0, merge_invariant=True,
    sphere_steps=20, sphere_loss=25,
)
# ---------------------------- optimizers/shampoo ---------------------------- #
Shampoo = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.GraftModules(tz.m.Shampoo(), tz.m.RMSprop()), tz.m.LR(4)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.GraftModules(tz.m.Shampoo(), tz.m.RMSprop()), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=0.02, merge_invariant=False,
    sphere_steps=20, sphere_loss=1, # merge and unmerge lrs are very different so need to test convergence separately somewhere
)

# ------------------------- quasi_newton/quasi_newton ------------------------ #
BFGS = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.BFGS(ptol_reset=True), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.BFGS(ptol_reset=True), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=50, loss=1e-10, merge_invariant=True,
    sphere_steps=10, sphere_loss=1e-10,
)
SR1 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.SR1(ptol_reset=True), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.SR1(ptol_reset=True), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=50, loss=1e-12, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)
SSVM = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.SSVM(1, ptol_reset=True), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.SSVM(1, ptol_reset=True), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=50, loss=0.5, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

# ---------------------------- quasi_newton/lbfgs ---------------------------- #
LBFGS = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LBFGS(), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LBFGS(), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=50, loss=0, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

# ----------------------------- quasi_newton/lsr1 ---------------------------- #
LSR1 = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LSR1(scale_second=True), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LSR1(scale_second=True), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=50, loss=0, merge_invariant=True,
    sphere_steps=10, sphere_loss=0,
)

# # ---------------------------- quasi_newton/olbfgs --------------------------- #
# OnlineLBFGS = Run(
#     func_opt=lambda p: tz.Modular(p, tz.m.OnlineLBFGS(), tz.m.StrongWolfe()),
#     sphere_opt=lambda p: tz.Modular(p, tz.m.OnlineLBFGS(), tz.m.StrongWolfe()),
#     needs_closure=True,
#     func='rosen', steps=50, loss=0, merge_invariant=True,
#     sphere_steps=10, sphere_loss=0,
# )

# ---------------------------- second_order/newton --------------------------- #
Newton = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Newton(), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Newton(), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=20, loss=1e-7, merge_invariant=True,
    sphere_steps=2, sphere_loss=1e-9,
)

# --------------------------- second_order/newton_cg -------------------------- #
NewtonCG = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.NewtonCG(), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.NewtonCG(), tz.m.StrongWolfe()),
    needs_closure=True,
    func='rosen', steps=20, loss=1e-7, merge_invariant=True,
    sphere_steps=2, sphere_loss=3e-4,
)

# ---------------------------- smoothing/gaussian ---------------------------- #
GaussianHomotopy = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.GaussianHomotopy(10, 1, tol=1e-1, seed=0), tz.m.BFGS(ptol_reset=True), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.GaussianHomotopy(10, 1, tol=1e-1, seed=0), tz.m.BFGS(ptol_reset=True), tz.m.StrongWolfe()),
    needs_closure=True,
    func='booth', steps=20, loss=0.1, merge_invariant=True,
    sphere_steps=10, sphere_loss=200,
)

# ---------------------------- smoothing/laplacian --------------------------- #
LaplacianSmoothing = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LaplacianSmoothing(min_numel=1), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LaplacianSmoothing(min_numel=1), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=0.4, merge_invariant=False,
    sphere_steps=10, sphere_loss=3,
)

LaplacianSmoothing_global = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LaplacianSmoothing(layerwise=False), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LaplacianSmoothing(layerwise=False), tz.m.LR(0.5)),
    needs_closure=False,
    func='booth', steps=50, loss=0.4, merge_invariant=True,
    sphere_steps=10, sphere_loss=3,
)

# -------------------------- wrappers/optim_wrapper -------------------------- #
Wrap = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Wrap(torch.optim.Adam, lr=1), tz.m.LR(0.5)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Wrap(torch.optim.Adam, lr=1), tz.m.LR(0.2)),
    needs_closure=False,
    func='rosen', steps=50, loss=4, merge_invariant=True,
    sphere_steps=20, sphere_loss=4,
)

# --------------------------- second_order/nystrom --------------------------- #
NystromSketchAndSolve = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.NystromSketchAndSolve(2, seed=0), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.NystromSketchAndSolve(10, seed=0), tz.m.StrongWolfe()),
    needs_closure=True,
    func='booth', steps=3, loss=1e-6, merge_invariant=True,
    sphere_steps=10, sphere_loss=1e-12,
)
NystromPCG = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.NystromPCG(2, seed=0), tz.m.StrongWolfe()),
    sphere_opt=lambda p: tz.Modular(p, tz.m.NystromPCG(10, seed=0), tz.m.StrongWolfe()),
    needs_closure=True,
    func='ill', steps=2, loss=1e-5, merge_invariant=True,
    sphere_steps=2, sphere_loss=1e-9,
)

# ---------------------------- optimizers/sophia_h --------------------------- #
SophiaH = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.SophiaH(seed=0), tz.m.LR(0.1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.SophiaH(seed=0), tz.m.LR(0.3)),
    needs_closure=True,
    func='ill', steps=50, loss=0.02, merge_invariant=True,
    sphere_steps=10, sphere_loss=40,
)

# -------------------------- higher_order ------------------------- #
HigherOrderNewton = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.HigherOrderNewton(trust_method=None)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.HigherOrderNewton(2, trust_method=None)),
    needs_closure=True,
    func='rosen', steps=1, loss=2e-10, merge_invariant=True,
    sphere_steps=1, sphere_loss=1e-10,
)

# ---------------------------- optimizers/ladagrad --------------------------- #
LMAdagrad = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.LMAdagrad(), tz.m.LR(4)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.LMAdagrad(), tz.m.LR(5)),
    needs_closure=False,
    func='booth', steps=50, loss=1e-6, merge_invariant=True,
    sphere_steps=20, sphere_loss=1e-9,
)

# ------------------------------ optimizers/adan ----------------------------- #
Adan = Run(
    func_opt=lambda p: tz.Modular(p, tz.m.Adan(), tz.m.LR(1)),
    sphere_opt=lambda p: tz.Modular(p, tz.m.Adan(), tz.m.LR(0.1)),
    needs_closure=False,
    func='booth', steps=50, loss=60, merge_invariant=True,
    sphere_steps=20, sphere_loss=60,
)

# ------------------------------------ CGs ----------------------------------- #
for CG in (tz.m.PolakRibiere, tz.m.FletcherReeves, tz.m.HestenesStiefel, tz.m.DaiYuan, tz.m.LiuStorey, tz.m.ConjugateDescent, tz.m.HagerZhang, tz.m.HybridHS_DY, tz.m.ProjectedGradientMethod):
    for func_steps,sphere_steps_ in ([3,2], [10,10]): # CG should converge on 2D quadratic after 2nd step
        # but also test 10 to make sure it doesn't explode after converging
        Run(
            func_opt=lambda p: tz.Modular(p, CG(), tz.m.StrongWolfe(c2=0.1)),
            sphere_opt=lambda p: tz.Modular(p, CG(), tz.m.StrongWolfe(c2=0.1)),
            needs_closure=True,
            func='lstsq', steps=func_steps, loss=1e-10, merge_invariant=False, # strong wolfe adds float imprecision
            sphere_steps=sphere_steps_, sphere_loss=0,
        )

# ------------------------------- QN stability ------------------------------- #
# stability test
for QN in (
    tz.m.BFGS,
    partial(tz.m.BFGS, inverse=False),
    tz.m.SR1,
    partial(tz.m.SR1, inverse=False),
    tz.m.DFP,
    partial(tz.m.DFP, inverse=False),
    tz.m.BroydenGood,
    partial(tz.m.BroydenGood, inverse=False),
    tz.m.BroydenBad,
    partial(tz.m.BroydenBad, inverse=False),
    tz.m.Greenstadt1,
    tz.m.Greenstadt2,
    tz.m.ICUM,
    tz.m.ThomasOptimalMethod,
    tz.m.FletcherVMM,
    tz.m.Horisho,
    partial(tz.m.Horisho, inner=tz.m.GradientCorrection()),
    tz.m.Pearson,
    tz.m.ProjectedNewtonRaphson,
    tz.m.PSB,
    tz.m.McCormick,
    tz.m.SSVM,
):
    Run(
        func_opt=lambda p: tz.Modular(p, QN(scale_first=False, ptol_reset=True), tz.m.StrongWolfe()),
        sphere_opt=lambda p: tz.Modular(p, QN(scale_first=False, ptol_reset=True), tz.m.StrongWolfe()),
        needs_closure=True,
        func='lstsq', steps=50, loss=1e-10, merge_invariant=False,
        sphere_steps=10, sphere_loss=1e-20,
    )

# ---------------------------------------------------------------------------- #
#                                      run                                     #
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("run", RUNS)
def test_opt(run: Run): run.test()