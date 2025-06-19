from collections.abc import Callable, Sequence
import pytest
import torch
import torchzero as tz

def _booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

_BOOTH_X0 = torch.tensor([0., -8.])

def _get_trajectory(opt_fn: Callable, x0: torch.Tensor, merge: bool, use_closure: bool, steps: int):
    """Returns a Tensor - trajectory of `opt_fn` on the booth function."""
    trajectory = []
    if merge:
        params = x0.clone().requires_grad_()
        optimizer = opt_fn([params])
    else:
        params = [x0[0].clone().requires_grad_(), x0[1].clone().requires_grad_()]
        optimizer = opt_fn(params)

    for _ in range(steps):
        if use_closure:
            def closure(backward=True):
                trajectory.append(torch.stack([p.clone() for p in params]))

                loss = _booth(*params)
                if backward:
                    optimizer.zero_grad()
                    loss.backward()
                return loss

            loss = optimizer.step(closure)
            assert torch.isfinite(loss), f'non-finite loss {loss}'
            for p in params: assert torch.isfinite(p), f'non-finite params {params}'

        else:
            trajectory.append(torch.stack([p.clone() for p in params]))

            loss = _booth(*params)
            assert torch.isfinite(loss), f'non-finite loss {loss}'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for p in params: assert torch.isfinite(p), f'non-finite params {params}'


    return torch.stack(trajectory, 0), optimizer

def _compare_trajectories(opt1, t1:torch.Tensor, opt2, t2:torch.Tensor):
    assert torch.allclose(t1, t2, rtol=1e-4, atol=1e-6), f'trajectories dont match. opts:\n{opt1}\n{opt2}\ntrajectories:\n{t1}\n{t2}'

def _assert_identical_opts(opt_fns: Sequence[Callable], merge: bool, use_closure: bool, device, steps: int):
    """checks that all `opt_fns` have identical trajectories on booth"""
    x0 = _BOOTH_X0.clone().to(device=device)
    base_opt = None
    base_trajectory = None
    for opt_fn in opt_fns:
        t, opt = _get_trajectory(opt_fn, x0, merge, use_closure, steps)
        if base_trajectory is None or base_opt is None:
            base_trajectory = t
            base_opt = opt
        else: _compare_trajectories(base_opt, base_trajectory, opt, t)

def _assert_identical_merge(opt_fn: Callable, use_closure, device, steps: int):
    """checks that trajectories match with x and y parameters split and merged"""
    x0 = _BOOTH_X0.clone().to(device=device)
    merged, merged_opt = _get_trajectory(opt_fn, x0, merge=True, use_closure=use_closure, steps=steps)
    unmerged, unmerged_opt = _get_trajectory(opt_fn, x0, merge=False, use_closure=use_closure, steps=steps)
    _compare_trajectories(merged_opt, merged, unmerged_opt, unmerged)

def _assert_identical_closure(opt_fn: Callable, merge, device, steps: int):
    """checks that trajectories match  with and without closure"""
    x0 = _BOOTH_X0.clone().to(device=device)
    closure, closure_opt = _get_trajectory(opt_fn, x0, merge=merge, use_closure=True, steps=steps)
    no_closure, no_closure_opt = _get_trajectory(opt_fn, x0, merge=merge, use_closure=False, steps=steps)
    _compare_trajectories(closure_opt, closure, no_closure_opt, no_closure)

def _assert_identical_merge_closure(opt_fn: Callable, device, steps: int):
    """checks that trajectories match with x and y parameters split and merged and with and without closure"""
    x0 = _BOOTH_X0.clone().to(device=device)
    merge_closure, opt_merge_closure = _get_trajectory(opt_fn, x0, merge=True, use_closure=True, steps=steps)
    merge_no_closure, opt_merge_no_closure = _get_trajectory(opt_fn, x0, merge=True, use_closure=False, steps=steps)
    no_merge_closure, opt_no_merge_closure = _get_trajectory(opt_fn, x0, merge=False, use_closure=True, steps=steps)
    no_merge_no_closure, opt_no_merge_no_closure = _get_trajectory(opt_fn, x0, merge=False, use_closure=False, steps=steps)

    _compare_trajectories(opt_merge_closure, merge_closure, opt_merge_no_closure, merge_no_closure)
    _compare_trajectories(opt_merge_closure, merge_closure, opt_no_merge_closure, no_merge_closure)
    _compare_trajectories(opt_merge_closure, merge_closure, opt_no_merge_no_closure, no_merge_no_closure)

def _assert_identical_device(opt_fn: Callable, merge: bool, use_closure: bool, steps: int):
    """checks that trajectories match on cpu and cuda."""
    if not torch.cuda.is_available(): return
    cpu, cpu_opt = _get_trajectory(opt_fn, _BOOTH_X0.clone().cpu(), merge=merge, use_closure=use_closure, steps=steps)
    cuda, cuda_opt = _get_trajectory(opt_fn, _BOOTH_X0.clone().cuda(), merge=merge, use_closure=use_closure, steps=steps)
    _compare_trajectories(cpu_opt, cpu, cuda_opt, cuda.to(cpu))

@pytest.mark.parametrize('amsgrad', [True, False])
def test_adam(amsgrad):
    torch_fn = lambda p: torch.optim.Adam(p, lr=1, amsgrad=amsgrad)
    tz_fn = lambda p: tz.Modular(p, tz.m.Adam(amsgrad=amsgrad))
    tz_fn2 = lambda p: tz.Modular(p, tz.m.Adam(amsgrad=amsgrad), tz.m.LR(1)) # test LR fusing
    tz_fn3 = lambda p: tz.Modular(p, tz.m.Adam(amsgrad=amsgrad), tz.m.LR(1), tz.m.Add(1), tz.m.Sub(1))
    tz_fn4 = lambda p: tz.Modular(p, tz.m.Adam(amsgrad=amsgrad), tz.m.Add(1), tz.m.Sub(1), tz.m.LR(1))
    tz_fn5 = lambda p: tz.Modular(p, tz.m.Clone(), tz.m.Adam(amsgrad=amsgrad))
    tz_fn_ops = lambda p: tz.Modular(
        p,
        tz.m.DivModules(
            tz.m.EMA(0.9, debiased=True),
            [tz.m.SqrtEMASquared(0.999, debiased=True, amsgrad=amsgrad), tz.m.Add(1e-8)]
        ))
    tz_fn_ops2 = lambda p: tz.Modular(
        p,
        tz.m.DivModules(
            [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9)],
            [tz.m.EMASquared(0.999, amsgrad=amsgrad), tz.m.Sqrt(), tz.m.Debias2(beta=0.999), tz.m.Add(1e-8)]
        ))
    tz_fn_ops3 = lambda p: tz.Modular(
        p,
        tz.m.DivModules(
            [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9, beta2=0.999)],
            [tz.m.EMASquared(0.999, amsgrad=amsgrad), tz.m.Sqrt(), tz.m.Add(1e-8)]
        ))
    tz_fn_ops4 = lambda p: tz.Modular(
        p,
        tz.m.DivModules(
            [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9)],
            [
                tz.m.Pow(2),
                tz.m.EMA(0.999),
                tz.m.AccumulateMaximum() if amsgrad else tz.m.Identity(),
                tz.m.Sqrt(),
                tz.m.Debias2(beta=0.999),
                tz.m.Add(1e-8)]
        ))
    tz_fns = (torch_fn, tz_fn, tz_fn2, tz_fn3, tz_fn4, tz_fn5, tz_fn_ops, tz_fn_ops2, tz_fn_ops3, tz_fn_ops4)

    _assert_identical_opts(tz_fns, merge=True, use_closure=True, device='cpu', steps=10)
    for fn in tz_fns:
        _assert_identical_merge_closure(fn, device='cpu', steps=10)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=10)

@pytest.mark.parametrize('beta1', [0.5, 0.9])
@pytest.mark.parametrize('beta2', [0.99, 0.999])
@pytest.mark.parametrize('eps', [1e-1, 1e-8])
@pytest.mark.parametrize('amsgrad', [True, False])
@pytest.mark.parametrize('lr', [0.1, 1])
def test_adam_hyperparams(beta1, beta2, eps, amsgrad, lr):
    tz_fn = lambda p: tz.Modular(p, tz.m.Adam(beta1, beta2, eps, amsgrad=amsgrad), tz.m.LR(lr))
    tz_fn2 = lambda p: tz.Modular(p, tz.m.Adam(beta1, beta2, eps, amsgrad=amsgrad, alpha=lr))
    _assert_identical_opts([tz_fn, tz_fn2], merge=True, use_closure=True, device='cpu', steps=10)

@pytest.mark.parametrize('centered', [True, False])
def test_rmsprop(centered):
    torch_fn = lambda p: torch.optim.RMSprop(p, 1, centered=centered)
    tz_fn = lambda p: tz.Modular(p, tz.m.RMSprop(centered=centered, init='zeros'))
    tz_fn2 = lambda p: tz.Modular(
        p,
        tz.m.Div([tz.m.CenteredSqrtEMASquared(0.99) if centered else tz.m.SqrtEMASquared(0.99), tz.m.Add(1e-8)]),
    )
    tz_fn3 = lambda p: tz.Modular(
        p,
        tz.m.Div([tz.m.CenteredEMASquared(0.99) if centered else tz.m.EMASquared(0.99), tz.m.Sqrt(), tz.m.Add(1e-8)]),
    )
    tz_fns = (tz_fn, tz_fn2, tz_fn3)
    _assert_identical_opts([torch_fn, *tz_fns], merge=True, use_closure=True, device='cpu', steps=10)
    for fn in tz_fns:
        _assert_identical_merge_closure(fn, device='cpu', steps=10)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=10)


@pytest.mark.parametrize('beta', [0.5, 0.9])
@pytest.mark.parametrize('eps', [1e-1, 1e-8])
@pytest.mark.parametrize('centered', [True, False])
@pytest.mark.parametrize('lr', [0.1, 1])
def test_rmsprop_hyperparams(beta, eps, centered, lr):
    tz_fn = lambda p: tz.Modular(p, tz.m.RMSprop(beta, eps, centered, init='zeros'), tz.m.LR(lr))
    torch_fn = lambda p: torch.optim.RMSprop(p, lr, beta, eps=eps, centered=centered)
    _assert_identical_opts([torch_fn, tz_fn], merge=True, use_closure=True, device='cpu', steps=10)



@pytest.mark.parametrize('nplus', (1.2, 2))
@pytest.mark.parametrize('nminus', (0.5, 0.9))
@pytest.mark.parametrize('lb', [1e-8, 1])
@pytest.mark.parametrize('ub', [50, 1.5])
@pytest.mark.parametrize('lr', [0.1, 1])
def test_rprop(nplus, nminus, lb, ub, lr):
    tz_fn = lambda p: tz.Modular(p, tz.m.LR(lr), tz.m.Rprop(nplus, nminus, lb, ub, alpha=lr, backtrack=False))
    torch_fn = lambda p: torch.optim.Rprop(p, lr, (nminus, nplus), (lb, ub))
    _assert_identical_opts([torch_fn, tz_fn], merge=True, use_closure=True, device='cpu', steps=30)
    _assert_identical_merge_closure(tz_fn, 'cpu', 30)
    _assert_identical_device(tz_fn, merge=True, use_closure=True, steps=10)

def test_adagrad():
    torch_fn = lambda p: torch.optim.Adagrad(p, 1)
    tz_fn = lambda p: tz.Modular(p, tz.m.Adagrad(), tz.m.LR(1))
    tz_fn2 = lambda p: tz.Modular(
        p,
        tz.m.Div([tz.m.Pow(2), tz.m.AccumulateSum(), tz.m.Sqrt(), tz.m.Add(1e-10)]),
    )

    tz_fns = (tz_fn, tz_fn2)
    _assert_identical_opts([torch_fn, *tz_fns], merge=True, use_closure=True, device='cpu', steps=10)
    for fn in tz_fns:
        _assert_identical_merge_closure(fn, device='cpu', steps=10)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=10)



@pytest.mark.parametrize('initial_accumulator_value', [0, 1])
@pytest.mark.parametrize('eps', [1e-2, 1e-10])
@pytest.mark.parametrize('lr', [0.1, 1])
def test_adagrad_hyperparams(initial_accumulator_value, eps, lr):
    torch_fn = lambda p: torch.optim.Adagrad(p, lr, initial_accumulator_value=initial_accumulator_value, eps=eps)
    tz_fn1 = lambda p: tz.Modular(p, tz.m.Adagrad(initial_accumulator_value=initial_accumulator_value, eps=eps), tz.m.LR(lr))
    tz_fn2 = lambda p: tz.Modular(p, tz.m.Adagrad(initial_accumulator_value=initial_accumulator_value, eps=eps, alpha=lr))
    _assert_identical_opts([torch_fn, tz_fn1, tz_fn2], merge=True, use_closure=True, device='cpu', steps=10)


@pytest.mark.parametrize('tensorwise', [True, False])
def test_graft(tensorwise):
    graft1 = lambda p: tz.Modular(p, tz.m.GraftModules(tz.m.LBFGS(), tz.m.RMSprop(), tensorwise=tensorwise), tz.m.LR(1e-1))
    graft2 = lambda p: tz.Modular(p, tz.m.LBFGS(), tz.m.Graft([tz.m.Grad(), tz.m.RMSprop()], tensorwise=tensorwise), tz.m.LR(1e-1))
    _assert_identical_opts([graft1, graft2], merge=True, use_closure=True, device='cpu', steps=10)
    for fn in [graft1, graft2]:
        if tensorwise: _assert_identical_closure(fn, merge=True, device='cpu', steps=10)
        else: _assert_identical_merge_closure(fn, device='cpu', steps=10)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=10)

