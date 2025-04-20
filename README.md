![example workflow](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)

note - I've recently revamped the internals to be significantly cleaner and allow for projections (e.g. GaLore, GWT), new docs are WIP but should be finished soon after I re-add all modules.

# torchzero

`torchzero` implements a large number of chainable optimization modules that can be chained together to create custom optimizers:

```py
import torchzero as tz

optimizer = tz.Modular(
    model.parameters(),
    tz.m.Adam(),
    tz.m.Cautious(),
    tz.m.LR(1e-3),
    tz.m.WeightDecay(1e-4)
)

# standard training loop
for batch in dataset:
    preds = model(batch)
    loss = criterion(preds)
    optimizer.zero_grad()
    optimizer.step()
```

Each module takes the output of the previous module and applies a further transformation. This modular design avoids redundant code, such as reimplementing cautioning, orthogonalization, laplacian smoothing, etc for every optimizer. It is also easy to experiment with grafting, interpolation between different optimizers, and perhaps some weirder combinations like nested momentum.

Modules are not limited to gradient transformations. They can perform other operations like line searches, exponential moving average (EMA) and stochastic weight averaging (SWA), gradient accumulation, gradient approximation, and more.

There are over 100 modules, all accessible within the `tz.m` namespace. For example, the Adam update rule is available as `tz.m.Adam`. Complete list of modules is available in [documentation](https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html).
<!--
## Closure

Some modules and optimizers in torchzero, particularly line-search methods and gradient approximation modules, require a closure function. This is similar to how `torch.optim.LBFGS` works in PyTorch. In torchzero, closure needs to accept a boolean backward argument (though the argument can have any name). When `backward=True`, the closure should zero out old gradients using `opt.zero_grad()`, and compute new gradients using `loss.backward()`.

```py
def closure(backward = True):
    preds = model(inputs)
    loss = loss_fn(preds, targets)

    if backward:
        optimizer.zero_grad()
        loss.backward()
    return loss

optimizer.step(closure)
```

If you intend to use gradient-free methods, `backward` argument is still required in the closure. Simply leave it unused. Gradient-free and gradient approximation methods always call closure with `backward=False`.

All built-in pytorch optimizers, as well as most custom ones, support closure too. So the code above will work with all other optimizers out of the box, and you can switch between different optimizers without rewriting your training loop. -->

## Installation

<!-- ```py
pip install torchzero
``` -->

## Documentation

<!-- Docs are heavily outdated. [torchzero.readthedocs.io](https://torchzero.readthedocs.io/en/latest/index.html). -->
<!--
# Extra

Some other optimization related things in torchzero:

### scipy.optimize.minimize wrapper

scipy.optimize.minimize wrapper with support for both gradient and hessian via batched autograd

```py
from torchzero.optim.wrappers.scipy import ScipyMinimize
opt = ScipyMinimize(model.parameters(), method = 'trust-krylov')
```

Use as any other closure-based optimizer, but make sure closure accepts `backward` argument. Note that it performs full minimization on each step.

### Nevergrad wrapper

[Nevergrad](https://github.com/facebookresearch/nevergrad) is an optimization library by facebook with an insane number of gradient free methods.

```py
from torchzero.optim.wrappers.nevergrad import NevergradOptimizer
opt = NevergradOptimizer(bench.parameters(), ng.optimizers.NGOptBase, budget = 1000)
```

Use as any other closure-based optimizer, but make sure closure accepts `backward` argument.

### NLopt wrapper

[NLopt](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) is another optimization library similar to scipy.optimize.minimize, with a large number of both gradient based and gradient free methods.

```py
from torchzero.optim.wrappers.nlopt import NLOptOptimizer
opt = NLOptOptimizer(bench.parameters(), 'LD_TNEWTON_PRECOND_RESTART', maxeval = 1000)
```

Use as any other closure-based optimizer, but make sure closure accepts `backward` argument. Note that it performs full minimization on each step. -->
