![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)

<h1 align='center'>torchzero</h1>

torchzero provides efficient implementations of 300+ optimization algorithms with pytorch optimizer interface, encompassing many classes of unconstrained optimization - convex and non-convex, local and global, derivative free, gradient based and second order, least squares, etc.

The algorithms are designed to be as modular as possibe - they can be freely combined, for example all second order-like methods can be combined with any line search or trust region algorithm. Techniques like gradient clipping, weight decay, sharpness-aware minimization, cautious updates, gradient accumulation are their own modules and can be used with anything else.

> **note:** This project is being actively developed, there may be API changes, although at this point I am very happy with the API.

## Installation

```bash
pip install torchzero
```

The github version may be a bit more recent and less tested:

```bash
pip install git+https://github.com/inikishev/torchzero
```

## How to use

Each module represents a distinct step in the optimization process. A list of modules implemented in torchzero available on the [wiki](https://inikishev.github.io/torchzero/API/).

Construct a ``tz.Modular`` optimizer with the desired modules and use as any other pytorch optimizer:

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

Certain modules such as line searches and trust regions require a closure, similar to L-BFGS in PyTorch. Also some modules require closure to accept an additional `backward` argument, refer to example below:

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

opt = tz.Modular([X], tz.m.NewtonCGSteihaug())
for step in range(24):
    loss = opt.step(closure)
    print(f'{step} - {loss}')
```

## Learn more

To learn more about how to use torchzero check [Basics](<https://inikishev.github.io/torchzero/Basics/>).

An overview of optimization algorithms in torchzero along with visualizations, explanations and benchmarks is available in the [overview section](<https://inikishev.github.io/torchzero/overview/0.%20Introduction/>).

If you just want to see what algorithms are implemeted, check [API reference](<https://inikishev.github.io/torchzero/API/>).
