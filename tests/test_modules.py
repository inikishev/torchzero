"""Sanity check on booth function. All optimizers should converge."""
import pytest
import torch
import torchzero as tz

def booth(x,y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

x0 = (0, -8)

__pylance = torch.tensor(float('inf'))
PRINT_LOSSES = True

def _test_optimizer(lmbda, tol=1e-1):
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
    for i in range(100):
        loss = opt.step(closure)
        losses.append(loss)
        assert isinstance(loss, torch.Tensor), (i, type(loss), loss)
        assert torch.isfinite(loss), (i, loss)
        
    assert loss <= tol, (tol, loss, [i.detach().cpu().item() for i in losses])
    if PRINT_LOSSES: print(opt.__class__.__name__, loss.detach().cpu().item())
    
    
OPTS = [
    # -------------------------------- OPTIMIZERS -------------------------------- #
    lambda p: tz.optim.GD(p, 0.1), # this uses backtracking line search by default which is why its different from SGD
    lambda p: tz.optim.SGD(p, 0.1),
    lambda p: tz.optim.SGD(p, 0.01, 0.9),
    lambda p: tz.optim.SGD(p, 0.01, 0.9,  nesterov=True),
    lambda p: tz.optim.SignSGD(p, 0.2),
    lambda p: tz.optim.SignSGD(p, 0.025, momentum = 0.9, nesterov=True), # this uses kwargs
    lambda p: tz.optim.NormSGD(p, 0.25),
    lambda p: tz.optim.Adagrad(p, 10),
    lambda p: tz.optim.Rprop(p, 10),
    lambda p: tz.optim.RMSProp(p, 10),
    lambda p: tz.optim.Adam(p, 0.9),
    lambda p: tz.optim.Grams(p, 1),
    lambda p: tz.optim.Lion(p, 1e-1),
    lambda p: tz.optim.CautiousAdam(p, 1),
    lambda p: tz.optim.CautiousSGD(p, 1e-2),
    lambda p: tz.optim.CautiousLion(p, 1e-1),
    lambda p: tz.optim.DirectionalNewton(p, 1e-2),
    lambda p: tz.optim.ExactNewton(p, 1),
    lambda p: tz.optim.NestedNesterov(p, 0.01,),
    lambda p: tz.optim.RandomCoordinateMomentum(p, 5e-2, 0.5),
    lambda p: tz.optim.GradMin(p, 2e-2),
    lambda p: tz.optim.FDM(p, 5e-2),
    lambda p: tz.optim.FDMWrapper(torch.optim.LBFGS(p)),
    lambda p: tz.optim.NewtonFDM(p, 1),
    lambda p: tz.optim.SPSA(p, 3e-2, 1e-3),
    lambda p: tz.optim.RandomizedFDMWrapper(torch.optim.Adam(p, 1), 1e-2, n_samples=16),
    lambda p: tz.optim.RandomSubspaceNewtonFDM(p, 2),
    
    # ---------------------------------- MODULES --------------------------------- #
    lambda p: tz.optim.Modular(p, tz.m.ScipyMinimizeScalarLS()),
    lambda p: tz.optim.Modular(p, [tz.m.Grafting(tz.m.Lion(), tz.m.Adam()), tz.m.LR(2e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.SignGrafting(tz.m.Lion(), tz.m.Adam()), tz.m.LR(4e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.IntermoduleCautious(tz.m.Lion(), tz.m.Adam()), tz.m.LR(4e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Sum([tz.m.Lion(), tz.m.Adam()]), tz.m.LR(4e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Mean([tz.m.Lion(), tz.m.Adam()]), tz.m.LR(8e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Subtract(tz.m.Lion(), tz.m.Adam()), tz.m.LR(8e-2)]),
    lambda p: tz.optim.Modular(p, [tz.m.Interpolate(tz.m.Lion(), tz.m.Adam(), 0.5), tz.m.LR(8e-2)]),
    # note
    # gradient centralization and laplacian smoothing (and as I understand whitening if I add it)
    # will need to be tested separately as they won't work with 2 scalars.
]

@pytest.mark.parametrize('opt', OPTS)
def test_optimizer(opt):
    _test_optimizer(opt)