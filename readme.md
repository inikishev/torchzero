![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)

<h1 align='center'>torchzero</h1>

torchzero is a general purpose optimization library with a highly modular design. There are many algorithms implemented in torchzero, including first and second order algorithms, Quasi-Newton methods, conjugate gradient, gradient approximations, line searches, trust regions, etc. The modularity allows, for example, to use Newton or any Quasi-Newton method with any trust region or line search.

> **note:** This project is being actively developed, there may be API changes.

## How to use

An overview of the modules available in torchzero is available on the [docs](https://inikishev.github.io/torchzero/API/).

Construct a modular optimizer and use like any other pytorch optimizer, although some modules require a closure as detailed in the next section.

```py
optimizer = tz.Modular(
    model.parameters(),
    tz.m.ClipValue(1),
    tz.m.Adam(),
    tz.m.WeightDecay(1e-2),
    tz.m.LR(1e-1)
)
```

Here is what happens:

1. The gradient is passed to the ``ClipValue(1)`` module, which returns gradient with magnitudes clipped to be no larger than 1.

2. Clipped gradient is passed to ``Adam()``, which updates Adam momentum buffers and returns the Adam update.

3. The Adam update is passed to ``WeightDecay()`` which adds a weight decay penalty to the Adam update. Since we placed it after Adam, the weight decay is decoupled. By moving ``WeightDecay()`` before ``Adam()``, we can get coupled weight decay.

4. Finally the update is passed to ``LR(0.1)``, which multiplies it by the learning rate of 0.1.

## Advanced optimization

Certain modules, particularly line searches and gradient approximations require a closure, similar to L-BFGS in PyTorch. Also some modules require closure to accept an additional `backward` argument, refer to example below:

```python
model = nn.Sequential(nn.Linear(10, 10), nn.ELU(), nn.Linear(10, 1))
inputs = torch.randn(100,10)
targets = torch.randn(100, 1)

optimizer = tz.Modular(
    model.parameters(),
    tz.m.CubicRegularization(tz.m.Newton()),
)

for i in range(1, 51):

    def closure(backward=True):
        preds = model(inputs)
        loss = F.mse_loss(preds, targets)

        # If backward=True, closure should call
        # optimizer.zero_grad() and loss.backward()
        if backward:
            optimizer.zero_grad()
            loss.backward()

        return loss

    loss = optimizer.step(closure)

    if i % 10 == 0:
        print(f"step: {i}, loss: {loss.item():.4f}")
```

The code above will also work with any other optimizer because all PyTorch optimizers and most custom ones support closure, so there is no need to rewrite training loop.

Rosenbrock minimization example:

```py
import torch
import torchzero as tz

def rosen(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

X = torch.tensor([-1.1, 2.5], requires_grad=True)

def closure(backward=True):
    loss = rosen(*X)
    if backward:
        X.grad = None # same as opt.zero_grad()
        loss.backward()
    return loss

opt = tz.Modular([X], tz.m.NewtonCGSteihaug(hvp_method='forward'))
for step in range(24):
    loss = opt.step(closure)
    print(f'{step} - {loss}')
```

## Wiki

More information and examples along with visualizations and explanations of many of the algorithms implemented in torchzero are available on the [wiki](https://inikishev.github.io/torchzero/overview/Basics/)

## Installation

torchzero can be installed from The Python Package Index:

```bash
pip install torchzero
```

Alternatively install it directly from this repo:

```bash
pip install git+https://github.com/inikishev/torchzero
```
