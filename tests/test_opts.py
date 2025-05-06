"""snity tests to make sure everything works and converges on basic functions"""
from collections.abc import Callable
import torch
import torchzero as tz

def _booth(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def _rosen(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def _ill(x, y):
    return x**2 + y**2 + 1.99999*x*y

funcs = {"booth": (_booth,  (0, -8)), "rosen": (_rosen, (-1.1, 2.5)), "ill": (_ill, (-9, 2.5))}

class _TestModel(torch.nn.Module):
    """sphere with all kinds of parameter shapes"""
    def __init__(self):
        super().__init__()

        params = [
            torch.tensor(1.), torch.tensor([1.]), torch.tensor([[1.]]),
            torch.randn(10), torch.randn(1,10), torch.randn(10,1), torch.randn(1,1,10),torch.randn(1,10,1),torch.randn(1,1,10),
            torch.randn(10,10), torch.randn(4,4,4), torch.randn(3,3,3,3), torch.randn(2,2,2,2,2,2,2),
            torch.randn(10,1,3,1,1),
            torch.zeros(2,2), torch.ones(2,2),
        ]
        self.params = torch.nn.ParameterList(torch.nn.Parameter(t) for t in params)

    def forward(self):
        return torch.sum(torch.stack([p.sum() for p in self.params]))

def _run_objective(opt, objective: Callable, use_closure: bool, steps: int):
    losses = []
    for _ in range(steps):
        if use_closure:
            def closure(backward=True):
                loss = objective()
                if backward:
                    opt.zero_grad()
                    loss.backward()
                return loss
            loss = opt.step(closure)
            assert loss is not None
            losses.append(loss)

        else:
            loss = objective()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss)

    return torch.stack(losses).nan_to_num(0,10000,10000).min()

def _run_func(opt_fn: Callable, func:str, merge: bool, use_closure: bool, steps: int):
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

    return _run_objective(opt, objective, use_closure, steps)

def _run_sphere(opt_fn: Callable, use_closure:bool, steps:int):
    sphere = _TestModel()
    opt = opt_fn(sphere.parameters())
    return _run_objective(opt, sphere, use_closure, steps)

def _run(opt_fn: Callable, needs_closure: bool, func:str, steps: int, loss: float, sphere_steps: int, sphere_loss: float):
    for merge in [True, False]:
        for use_closure in [True] if needs_closure else [True, False]:
            v = _run_func(opt_fn, func, merge, use_closure, steps)
            assert v <= loss

            v = _run_sphere(opt_fn, use_closure, sphere_steps)
            assert v <= sphere_loss


class Run:
    def __init__(self, opt_fn: Callable, needs_closure: bool, func: str, steps: int, loss:float, sphere_steps:int, sphere_loss:float):
        self.kwargs = locals().copy()
        del self.kwargs['self']
    def test(self): _run(**self.kwargs)

# ----------------------------------- tests ---------------------------------- #
adam = Run(
    opt_fn=lambda p: tz.Modular(p, tz.m.Adam()),
    needs_closure=False,
    func='rosen', steps=100, loss=1,
    sphere_steps=20, sphere_loss=1,
)

# ------------------------------------ run ----------------------------------- #
def test_opts():
    for v in globals().copy().values():
        if isinstance(v, Run):
            v.test()