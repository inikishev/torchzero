from typing import Any
import pytest
import torch
import torchzero as tz

def _func(x,abc):
    a,b,c = abc
    return (a*torch.sin(x) + b*torch.log(x.clamp(0.1))) ** 2 + c*x**2

def _test_against_reference(opt, reference_opt, n_steps = 10, dtype=torch.float32, device='cpu'):
    """test that opt and reference opt do the same thing.

    .. code-block:: python
        test_against_reference(
            lambda p: tz.optim.Modular(p, tz.m.Adam(1e-3)),
            lambda p: torch.optim.Adam(p, 1e-3)
        )
    """
    torch.use_deterministic_algorithms(True)
    history_opt = []
    history_reference = []

    coeffs_real = torch.tensor([1, 2, 3], dtype=dtype, device = device)
    inputs = torch.arange(-1, 1, 0.1, dtype=dtype, device = device)
    y_real = _func(inputs, coeffs_real)
    for opt_cls, history in (((opt, history_opt), (reference_opt, history_reference))):
        torch.manual_seed(0)

        coeffs_preds = torch.tensor([1.1, 1.9, 2.8], dtype=dtype, device = device, requires_grad=True)
        optimizer = opt_cls([coeffs_preds])

        def closure(backward=True):
            preds = _func(inputs, coeffs_preds)
            loss = (preds - y_real).pow(2).mean()
            if backward:
                optimizer.zero_grad()
                loss.backward()
            return loss

        for _ in range(n_steps):
            loss = optimizer.step(closure)
            history.append(coeffs_preds.detach().cpu())

    assert torch.allclose(torch.stack(history_opt), torch.stack(history_reference)), f"Optimizers do not match at {dtype=}, {device=}: \n{history_opt=}\n{history_reference=}"


def test_test_against_reference():
    """test if _test_against_reference works"""
    _test_against_reference(
        lambda p: tz.optim.Modular(p, tz.m.Adam(1e-3)),
        lambda p: torch.optim.Adam(p, 1e-3),
        n_steps = 10,
    )

    with pytest.raises(AssertionError):
        _test_against_reference(
            lambda p: tz.optim.Modular(p, tz.m.Adam(1e-3)),
            lambda p: torch.optim.Adam(p, 0.99e-3),
            n_steps = 10,
        )


@pytest.mark.parametrize('lr', [1e-3, 1e-4])
@pytest.mark.parametrize('momentum', [0, 0.9])
@pytest.mark.parametrize('dampening', [0, 0.9])
@pytest.mark.parametrize('weight_decay', [0, 0.1])
@pytest.mark.parametrize('nesterov', [True, False])
def test_sgd(lr, momentum, dampening, weight_decay, nesterov):
    """torch.optim.SGD"""
    if nesterov:
        if dampening > 0: return # pytorch doesn't support nesterov with dampening
        if momentum == 0: return
    _test_against_reference(
        lambda p: tz.optim.Modular(p, tz.m.SGD(lr, momentum, dampening, weight_decay, nesterov)),
        lambda p: torch.optim.SGD(p, lr, momentum, dampening, weight_decay, nesterov),
    )

_devices = ['cpu']
if torch.cuda.is_available(): _devices.append('cuda')

# with adam we also test all dtypes and devices
@pytest.mark.parametrize('lr', [1e-1, 1e-4])
@pytest.mark.parametrize('betas', [(0.9, 0.999), (0.95, 0.95)])
@pytest.mark.parametrize('eps', [1e-8, 1e-4])
@pytest.mark.parametrize('amsgrad', [True, False])
@pytest.mark.parametrize('weight_decay', [0, 0.1])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', _devices)
def test_adam(lr, betas, eps, amsgrad, weight_decay, dtype, device):
    """torch.optim.Adam"""
    # note! it is necessary NOT to pre-initialize modules! because they will keep buffers and produce bogus results!
    if weight_decay > 0:
        _test_against_reference(
            lambda p: tz.optim.Modular(p, [tz.m.WeightDecay(weight_decay), tz.m.Adam(lr, *betas, eps=eps, amsgrad=amsgrad)]),
            lambda p: torch.optim.Adam(p, lr, betas, eps, amsgrad = amsgrad, weight_decay=weight_decay),
            dtype = dtype, device = device,

        )
    else:
        _test_against_reference(
            lambda p: tz.optim.Modular(p, tz.m.Adam(lr, *betas, eps=eps, amsgrad=amsgrad)),
            lambda p: torch.optim.Adam(p, lr, betas, eps, amsgrad = amsgrad, weight_decay=weight_decay),
            dtype = dtype, device = device,
        )


@pytest.mark.parametrize('lr', [1e-1, 1e-4])
@pytest.mark.parametrize('alpha', [0.9, 0.95])
@pytest.mark.parametrize('eps', [1e-8, 1e-4])
@pytest.mark.parametrize('centered', [True, False])
def test_rmsprop(lr, alpha, eps, centered):
    """torch.optim.RMSProp"""
    _test_against_reference(
        lambda p: tz.optim.Modular(p, [tz.m.RMSProp(alpha, eps, centered), tz.m.LR(lr)]),
        lambda p: torch.optim.RMSprop(p, lr, alpha, eps, centered = centered),
    )

@pytest.mark.parametrize('lr', [1e-1, 1e-4])
@pytest.mark.parametrize('etas', [(1.2, 0.5), (1.01, 0.99)])
@pytest.mark.parametrize('bounds', [(1e-8, 50), (2e-3, 5e-2)])
def test_rprop(lr, etas, bounds):
    """torch.optim.Rprop"""
    _test_against_reference(
        lambda p: tz.optim.Modular(p, tz.m.Rprop(lr, *etas, *bounds, backtrack = False)),
        lambda p: torch.optim.Rprop(p, lr, list(reversed(etas)), bounds), # type:ignore
    )


@pytest.mark.parametrize('lr', [1e-1, 1e-4])
@pytest.mark.parametrize('lr_decay', [0, 0.1])
@pytest.mark.parametrize('initial_accumulator_value', [0., 1.])
@pytest.mark.parametrize('eps', [1e-10, 1e-5])
def test_adagrad(lr, lr_decay, initial_accumulator_value, eps):
    """torch.optim.Rprop"""
    _test_against_reference(
        lambda p: tz.optim.Modular(p, tz.m.Adagrad(lr, lr_decay, initial_accumulator_value, eps)),
        lambda p: torch.optim.Adagrad(p, lr, lr_decay, initial_accumulator_value = initial_accumulator_value, eps = eps), # type:ignore
    )

@pytest.mark.parametrize('lr', [1e-1])
@pytest.mark.parametrize('compare', ['ascent', 'grad', tz.m.Mul(1)])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('mode', ['zero', 'grad', 'backtrack'])
@pytest.mark.parametrize('modular', [True, False])
def test_cautious_vs_intermodule(lr, compare,normalize, mode,modular):
    """tests IntermoduleCautious"""
    if modular: opt1 = lambda p: tz.optim.Modular(p, tz.m.Adam(lr), tz.m.Cautious(normalize=normalize, mode=mode))
    else: opt1 = lambda p: tz.optim.CautiousAdamW(p, lr, normalize=normalize, mode=mode)
    _test_against_reference(
        opt1,
        lambda p: tz.optim.Modular(p, tz.m.IntermoduleCautious(tz.m.Adam(lr), compare, normalize=normalize, mode=mode)), # type:ignore
    )