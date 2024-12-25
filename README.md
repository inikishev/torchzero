![example workflow](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)
# torchzero
This is a work-in-progress optimizers library for pytorch with composable zeroth, first, second order and quasi newton methods, gradient approximation, line searches and a whole lot of other stuff.

Most optimizers are modular, meaning you can chain them like this:
```py
optimizer = torchzero.optim.Modular(model.parameters(), [*list of modules*])`
```
For example you might use `[ClipNorm(4), LR(1e-3), NesterovMomentum(0.9)]` for standard SGD with gradient clipping and nesterov momentum. Move `ClipNorm` to the end to clip the update instead of the gradients. If you don't have access to gradients, add a `RandomizedFDM()` at the beginning to approximate them via randomized finite differences. Add `Cautious()` to make the optimizer cautious.

Each new module takes previous module update and works on it. That way there is no need to reimplement stuff like laplacian smoothing for all optimizers, and it is easy to experiment with grafting, interpolation between different optimizers, and perhaps some weirder combinations like nested momentum. 

# How to use

All modules are defined in `torchzero.modules`. You can generally mix and match them however you want. Some pre-made optimizers are available in `torchzero.optim`.

Some optimizers require closure, which should look like this:
```py
def closure(backward = True):
  preds = model(inputs)
  loss = loss_fn(preds, targets)

  # if you can't call loss.backward() and use gradient-free methods, they always call closure with backward=False.
  # so you can remove the part below, but keep the unused backward argument.
  if backward:
    optimizer.zero_grad()
    loss.backward()
  return loss

optimizer.step(closure)
```
This closure will also work with all built in pytorch optimizers, including LBFGS, all optimizers in this library, as well as most custom ones.

# Contents
There will be docs with a more exhaustive list and explanations. A preliminary list of all modules is available here https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html#classes. For now I hope that everything should be reasonably straightforward to use.
- SGD/Rprop/RMSProp/AdaGrad/Adam as composable modules. They are also tested to exactly match built in pytorch versions.
- Cautious Optimizers (https://huggingface.co/papers/2411.16085)
- Optimizer grafting (https://openreview.net/forum?id=FpKgG31Z_i9)
- Laplacian smoothing (https://arxiv.org/abs/1806.06317)
- Polyak momentum, nesterov momentum
- Gradient norm and value clipping, gradient normalization
- Gradient centralization (https://arxiv.org/abs/2004.01461)
- Learning rate droput (https://pubmed.ncbi.nlm.nih.gov/35286266/).
- Forward gradients (https://arxiv.org/abs/2202.08587)
- Gradient approximation via finite difference or randomized finite difference, which includes SPSA, RDSA, FDSA and Gaussian smoothing (https://arxiv.org/abs/2211.13566v3)
- Various line searches
- Exact Newton's method (with Levenberg-Marquardt regularization), newton with hessian approximation via finite difference, subspace finite differences newton.
- Directional newton via one additional forward pass

All modules should be quite fast, especially on models with many different parameters, due to `_foreach` operations.

I am getting to the point where I can start focusing on good docs and tests. As of now, the code should be considered experimental, untested and subject to change, so feel free but be careful if using this for actual project.


# Wrappers
### scipy.optimize.minimize wrapper
scipy.optimize.minimize wrapper with support for both gradient and hessian via batched autograd
```py
from torchzero.optim.wrappers.scipy import ScipyMinimize
opt = ScipyMinimize(model.parameters(), method = 'trust-krylov')
```
Use as any other optimizer (make sure closure accepts `backward` argument like one from **How to use**). Note that it performs full minimization on each step. 

### Nevergrad wrapper
```py
opt = NevergradOptimizer(bench.parameters(), ng.optimizers.NGOptBase, budget = 1000)
```
Use as any other optimizer (make sure closure accepts `backward` argument like one from **How to use**).
