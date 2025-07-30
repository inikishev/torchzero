![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)

<h1 align='center'>torchzero</h1>

torchzero is a general purpose optimization library with a highly modular design.

> **note:** This project is under development, API is subject to change and there may be bugs.

## What is done so far

There are A LOT of modules, including first order, quasi-newton, second order, conjugate gradient methods, line searches and trust regions, gradient approximations, gauss newton for least squares.

The list of modules is available here https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html, although it is slightly outdated since I decided to rewrite the wiki.

The modules represent gradient transformations and are freely combineable (see examples below). You can take newton, gauss-newton, any quasi-newton method, choose any line-search or trust region, add something else like restarts, even put a momentum or sharpness-aware minimization somewhere in the mix.

## How to use

Construct a modular optimizer and use like any other pytorch optimizer, although some modules require a closure as detailed in the next section.

```py
optimizer = tz.Modular(
    model.parameters(),
    tz.m.GradientAccumulation(4),
    tz.m.ClipValue(10),
    tz.m.Adam(),
    tz.m.NormalizeByEMA(max_ema_growth=1.1),
    tz.m.WeightDecay(1e-4),
    tz.m.LR(1e-1),
)
```

### Closure

Certain modules, particularly line searches and gradient approximations require a closure, similar to L-BFGS in PyTorch. Also some modules require closure to accept an additional `backward` argument, refer to example below:

```python
# training loop
for inputs, targets in dataloader:

    def closure(backward=True): # make sure it is True by default
        preds = model(inputs)
        loss = criterion(preds, targets)

        if backward: # gradient approximations always call with backward=False.
            optimizer.zero_grad()
            loss.backward()

        return loss

    loss = optimizer.step(closure)
```

The code above will also work with any other optimizer because all PyTorch optimizers and most custom ones support closure, so there is no need to rewrite training loop.

Non-batched example (rosenbrock):

```py
import torchzero as tz

def rosen(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

W = torch.tensor([-1.1, 2.5], requires_grad=True)

def closure(backward=True):
    loss = rosen(*W)
    if backward:
        W.grad = None # same as opt.zero_grad()
        loss.backward()
    return loss

opt = tz.Modular([W], tz.m.NewtonCGSteihaug())
for step in range(20):
    loss = opt.step(closure)
    print(f'{step} - {loss}')
```

## Wiki

The wiki is quite outdated https://torchzero.readthedocs.io/en/latest/index.html

## Installation

This will be available on pip once it is more refined. The latest version is available on the dev branch https://github.com/inikishev/torchzero/tree/dev.