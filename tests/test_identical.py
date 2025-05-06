from collections.abc import Callable, Sequence
import pytest
import torch
import torchzero as tz

def _booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

_BOOTH_X0 = torch.tensor([0., -8.])

def _get_trajectory(opt_fn: Callable, x0: torch.Tensor, merge: bool, use_closure: bool, steps: int):
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
    x0 = _BOOTH_X0.clone().to(device=device)
    base_opt = None
    base_trajectory = None
    for opt_fn in opt_fns:
        t, opt = _get_trajectory(opt_fn, x0, merge, use_closure, steps)
        if base_trajectory is None or base_opt is None:
            base_trajectory = t
            base_opt = opt
        else: _compare_trajectories(base_opt, base_trajectory, opt, t)

def _assert_identical_merge(opt_fn: Callable, device, steps: int):
    x0 = _BOOTH_X0.clone().to(device=device)
    merged, merged_opt = _get_trajectory(opt_fn, x0, merge=True, use_closure=True, steps=steps)
    unmerged, unmerged_opt = _get_trajectory(opt_fn, x0, merge=False, use_closure=True, steps=steps)
    _compare_trajectories(merged_opt, merged, unmerged_opt, unmerged)

def _assert_identical_merge_closure(opt_fn: Callable, device, steps: int):
    x0 = _BOOTH_X0.clone().to(device=device)
    merge_closure, opt_merge_closure = _get_trajectory(opt_fn, x0, merge=True, use_closure=True, steps=steps)
    merge_no_closure, opt_merge_no_closure = _get_trajectory(opt_fn, x0, merge=True, use_closure=False, steps=steps)
    no_merge_closure, opt_no_merge_closure = _get_trajectory(opt_fn, x0, merge=False, use_closure=True, steps=steps)
    no_merge_no_closure, opt_no_merge_no_closure = _get_trajectory(opt_fn, x0, merge=False, use_closure=False, steps=steps)

    _compare_trajectories(opt_merge_closure, merge_closure, opt_merge_no_closure, merge_no_closure)
    _compare_trajectories(opt_merge_closure, merge_closure, opt_no_merge_closure, no_merge_closure)
    _compare_trajectories(opt_merge_closure, merge_closure, opt_no_merge_no_closure, no_merge_no_closure)

def _assert_identical_device(opt_fn: Callable, merge: bool, use_closure: bool, steps: int):
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
            [tz.m.EMA(0.9), tz.m.Debias1(0.9)],
            [tz.m.EMASquared(0.999, amsgrad=amsgrad), tz.m.Sqrt(), tz.m.Debias2(0.999), tz.m.Add(1e-8)]
        ))
    tz_fn_ops3 = lambda p: tz.Modular(
        p,
        tz.m.DivModules(
            [tz.m.EMA(0.9), tz.m.Debias1(0.9)],
            [
                tz.m.Pow(2),
                tz.m.EMA(0.999),
                tz.m.AccumulateMaximum() if amsgrad else tz.m.Identity(),
                tz.m.Sqrt(),
                tz.m.Debias2(0.999),
                tz.m.Add(1e-8)]
        ))
    tz_fns = (tz_fn, tz_fn2, tz_fn3, tz_fn4, tz_fn5, tz_fn_ops, tz_fn_ops2, tz_fn_ops3)

    _assert_identical_opts([torch_fn, *tz_fns], merge=True, use_closure=True, device='cpu', steps=100)
    for fn in tz_fns:
        _assert_identical_merge_closure(fn, device='cpu', steps=100)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=100)

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
    _assert_identical_opts([torch_fn, *tz_fns], merge=True, use_closure=True, device='cpu', steps=100)
    for fn in tz_fns:
        _assert_identical_merge_closure(fn, device='cpu', steps=100)
        _assert_identical_device(fn, merge=True, use_closure=True, steps=100)
