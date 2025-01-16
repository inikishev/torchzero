"""Sanity check on booth function. All optimizers should converge."""
import importlib.util

import numpy as np
import pytest
import torch
import torchzero as tz

PRINT_LOSSES = False

def booth(x,y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

x0 = (0, -8)

__pylance = torch.tensor(float('inf'))

def _ensure_float(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu().item()
    if isinstance(x, np.ndarray): return x.item()
    return x

def _test_optimizer(lmbda, tol=1e-1, niter=100, allow_non_tensor=False):
    params = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    opt = lmbda([params])

    def closure(backward=True):
        loss = booth(*params)
        if backward:
            opt.zero_grad()
            loss.backward()
        return loss

    loss = __pylance
    losses = []
    for i in range(niter):
        loss = opt.step(closure)
        losses.append(loss)

        if allow_non_tensor:
            assert isinstance(loss, (torch.Tensor, np.ndarray, int, float)), (opt.__class__.__name__, i, type(loss), loss)
        else:
            assert isinstance(loss, torch.Tensor), (opt.__class__.__name__, i, type(loss), loss)

        if isinstance(loss, torch.Tensor): assert torch.isfinite(loss), (opt.__class__.__name__, i, loss)
        else: assert np.isfinite(loss), (opt.__class__.__name__, i, loss)

    assert loss <= tol, (
        opt.__class__.__name__,
        [i.__class__.__name__ for i in opt.unrolled_modules] if hasattr(opt, 'unrolled_modules') else None,
        tol,
        loss,
        [i.detach().cpu().item() for i in losses]
    )

    if PRINT_LOSSES:
        print(
            opt.__class__.__name__,
            [i.__class__.__name__ for i in opt.unrolled_modules] if hasattr(opt, 'unrolled_modules') else None,
            _ensure_float(loss)
        )

OPTS = [
    # -------------------------------- OPTIMIZERS -------------------------------- #
    lambda p: tz.optim.GD(p, 0.1), # this uses armijo line search by default which is why its different from SGD
    lambda p: tz.optim.SGD(p, 0.1),
    lambda p: tz.optim.SGD(p, 0.01, 0.9),
    lambda p: tz.optim.SGD(p, 0.01, 0.9,  nesterov=True),
    lambda p: tz.optim.SignSGD(p, 0.2),
    lambda p: tz.optim.SignSGD(p, 0.025, momentum = 0.9, nesterov=True), # this uses kwargs
    lambda p: tz.optim.NormSGD(p, 0.25),
    lambda p: tz.optim.Adagrad(p, 10),
    lambda p: tz.optim.Rprop(p, 10),
    lambda p: tz.optim.RMSProp(p, 10),
    lambda p: tz.optim.AdamW(p, 0.9),
    lambda p: tz.optim.Grams(p, 1),
    lambda p: tz.optim.Lion(p, 1),
    lambda p: tz.optim.CautiousAdamW(p, 1),
    lambda p: tz.optim.CautiousSGD(p, 1e-2),
    lambda p: tz.optim.CautiousLion(p, 1),
    lambda p: tz.optim.DirectionalNewton(p, 1e-2),
    lambda p: tz.optim.ExactNewton(p, 1),
    lambda p: tz.optim.NestedNesterov(p, 0.01,),
    lambda p: tz.optim.experimental.RandomCoordinateMomentum(p, 5e-2, 0.5),
    lambda p: tz.optim.experimental.GradMin(p, 2e-2),
    lambda p: tz.optim.FDM(p, 5e-2),
    lambda p: tz.optim.NewtonFDM(p, 1),
    lambda p: tz.optim.SPSA(p, 3e-2, 1e-3),
    lambda p: tz.optim.FDMWrapper(torch.optim.LBFGS(p)),
    lambda p: tz.optim.RandomizedFDMWrapper(torch.optim.Adam(p, 1), 1e-2, n_samples=16),
    lambda p: tz.optim.RandomSubspaceNewtonFDM(p, 2),

    # ---------------------------------- MODULES --------------------------------- #
    lambda p: tz.optim.Modular(p, tz.m.ScipyMinimizeScalarLS()),
    lambda p: tz.optim.Modular(p, [tz.m.Graft(tz.m.Grad(), tz.m.Adam()), tz.m.LR(2e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.SignGrafting(tz.m.Lion(), tz.m.Adam()), tz.m.LR(3e-1)]),
    lambda p: tz.optim.Modular(p, [tz.m.IntermoduleCautious(tz.m.Lion(), tz.m.Adam()), tz.m.LR(5e-1)]),
    lambda p: tz.optim.Modular(p, [tz.m.Sum(tz.m.Rprop(), tz.m.Adam()), tz.m.LR(1e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Mean(tz.m.Rprop(), tz.m.Adam()), tz.m.LR(2e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Subtract(tz.m.Rprop(), tz.m.Adam(alpha=20)), tz.m.LR(1e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Interpolate(tz.m.Rprop(), tz.m.Adam(), 0.5), tz.m.LR(3e-1)]),
    lambda p: tz.optim.Modular(p, tz.m.experimental.MinibatchRprop(), tz.m.LR(1e-1)),
    # note
    # gradient centralization and laplacian smoothing (and as I understand whitening if I add it)
    # will need to be tested separately as they won't work with 2 scalars.
]

@pytest.mark.parametrize('opt', OPTS)
def test_optimizer(opt):
    _test_optimizer(opt)


def test_scipy_wrapper():
    from torchzero.optim.wrappers.scipy import ScipyMinimize
    _test_optimizer(ScipyMinimize, niter=1, allow_non_tensor=True)

def test_nevergrad_wrapper():
    if importlib.util.find_spec('nevergrad') is not None:
        import nevergrad as ng
        from torchzero.optim.wrappers.nevergrad import NevergradOptimizer
        _test_optimizer(lambda p: NevergradOptimizer(p, ng.optimizers.OnePlusOne), niter=500, allow_non_tensor=True)

@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute")
@pytest.mark.filterwarnings("ignore:making a non-writeable array writeable is deprecated for arrays without a base which do not own their data.")
def test_nlopt_wrapper():
    if importlib.util.find_spec('nlopt') is not None:
        from torchzero.optim.wrappers.nlopt import NLOptOptimizer
        _test_optimizer(lambda p: NLOptOptimizer(p, 'LN_BOBYQA', 100), niter=1, allow_non_tensor=True)