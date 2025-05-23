# torchzero

**Modular optimization library for PyTorch**

<!-- [![PyPI version](https://img.shields.io/pypi/v/torchzero.svg)](https://pypi.org/project/torchzero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/actions/workflow/status/torchzero/torchzero/ci.yml?branch=main)](https://github.com/torchzero/torchzero/actions)
[![Documentation Status](https://readthedocs.org/projects/torchzero/badge/?version=latest)](https://torchzero.readthedocs.io/en/latest/?badge=latest) -->

`torchzero` is a Python library providing a highly modular framework for creating and experimenting with optimization algorithms in PyTorch. It allows users to easily combine and customize various components of optimizers, such as momentum techniques, gradient clipping, line searches and more.

NOTE: torchzero is in active development, currently docs are in a state of flux and pip version is extremely outdated.

## Installation

```bash
pip install git+https://github.com/inikishev/torchzero
```

(please don't use pip version yet, it is very outdated)

**Dependencies:**

* Python >= 3.10
* `torch`
* `numpy`
* `typing_extensions`

## Core Concepts

<!-- ### Modular Design

`torchzero` is built around a few key abstractions:

* **`Module`**: The base class for all components in `torchzero`. Each `Module` implements a `step(vars)` method that processes the optimization variables.
* **`Modular`**: The main optimizer class that chains together a sequence of `Module`s. It orchestrates the flow of data through the modules in the order they are provided.
* **`Transform`**: A special type of `Module` designed for tensor transformations. These are often used for operations like applying momentum or scaling gradients.
* **`Preconditioner`**: A subclass of `Transform`, typically used for preconditioning gradients (e.g., Adam, RMSprop).

### `Vars` Object

The `Vars` object is a data carrier that passes essential information between modules during an optimization step. It typically holds:

* `params`: The model parameters.
* `grad`: Gradients of the parameters.
* `update`: The update to be applied to the parameters.
* `loss`: The current loss value.
* `closure`: A function to re-evaluate the model and loss (used by some line search algorithms and other modules that might need to recompute gradients or loss).

### `TensorList`

`torchzero` uses a custom `TensorList` class for efficient batched operations on lists of tensors. This allows for optimized performance when dealing with multiple parameter groups or complex update rules. -->

## Quick Start / Usage Example

Here's a basic example of how to use `torchzero`:

```python
import torch
from torch import nn
import torchzero as tz

# Define a simple model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
inputs = torch.randn(5, 10)
targets = torch.randn(5, 1)

# Create an optimizer
# The order of modules matters:
# 1. SOAP: Computes the update.
# 2. NormalizeByEMA: stabilizes the update by normalizing to an exponential moving average of past updates.
# 3. WeightDecay - semi-decoupled, because it is applied after SOAP, but before LR
# 4. LR: Scales the computed update by the learning rate (supports LR schedulers).
optimizer = tz.Modular(
    model.parameters(),
    tz.m.SOAP(),
    tz.m.NormalizeByEMA(max_ema_growth=1.1),
    tz.m.WeightDecay(1e-4),
    tz.m.LR(1e-1),
)

# Standard training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## Overview of Available Modules

`torchzero` provides a rich set of pre-built modules. Here are some key categories and examples:

* **Optimizers (`torchzero/modules/optimizers/`)**: Optimization algorithms.
  * `Adam`.
  * `Shampoo`.
  * `SOAP` (my current recommendation).
  * `Muon`.
  * `SophiaH`.
  * `Adagrad` and `FullMatrixAdagrad`.
  * `Lion`.
  * `RMSprop`.
  * `OrthoGrad`.
  * `Rprop`.

  Additionally many other optimizers can be easily defined via modules:
  * Grams: `[tz.m.Adam(), tz.m.GradSign()]`
  * LaProp: `[tz.m.RMSprop(), tz.m.EMA(0.9)]`
  * Signum: `[tz.m.HeavyBall(), tz.m.Sign()]`
  * Full matrix version of any diagonal optimizer, like Adam: `tz.m.FullMatrixAdagrad(beta=0.999, inner=tz.m.EMA(0.9))`
  * Cautious version of any optimizer, like SOAP: `[tz.m.SOAP(), tz.m.Cautious()]`

* **Clipping (`torchzero/modules/clipping/`)**: Gradient clipping techniques.
  * `ClipNorm`: Clips gradient L2 norm.
  * `ClipValue`: Clips gradient values element-wise.
  * `Normalize`: Normalizes gradients to unit norm.
  * `Centralize`: Centralizes gradients by subtracting the mean.
  * `ClipNormByEMA`, `NormalizeByEMA`, `ClipValueByEMA`: Clipping/Normalization based on EMA of past values.
  * `ClipNormGrowth`, `ClipValueGrowth`: Limits norm or value growth.
* **Gradient Approximation (`torchzero/modules/grad_approximation/`)**: Methods for approximating gradients.
  * `FDM`: Finite Difference Method.
  * `RandomizedFDM` (`MeZO`, `SPSA`, `RDSA`, `Gaussian smoothing`): Randomized Finite Difference Methods (also subspaces).
  * `ForwardGradient`: Randomized gradient approximation via forward mode automatic differentiation.
* **Line Search (`torchzero/modules/line_search/`)**: Techniques for finding optimal step sizes.
  * `Backtracking`, `AdaptiveBacktracking`: Backtracking line searches.
  * `StrongWolfe`: Cubic interpolation line search satisfying strong Wolfe conditions.
  * `ScipyMinimizeScalar`: Wrapper for SciPy's scalar minimization for line search.
  * `TrustRegion`: First order trust region method.
* **Learning Rate (`torchzero/modules/lr/`)**: Learning rate control.
  * `LR`: Applies a fixed learning rate.
  * `PolyakStepSize`: Polyak's method.
  * `Warmup`: Learning rate warmup.
* **Momentum (`torchzero/modules/momentum/`)**: Momentum-based update modifications.
  * `NAG`: Nesterov Accelerated Gradient.
  * `HeavyBall`: Classic momentum (Polyak's momentum).
  * `EMA`: Exponential moving average.
  * `Averaging` (`Medianveraging`, `WeightedAveraging`): Simple, median, or weighted averaging of updates.
  * `Cautious`, `ScaleByGradCosineSimilarity`: Momentum cautioning.
  * `MatrixMomentum`, `AdaptiveMatrixMomentum`: Second order momentum.
  <!-- * `CoordinateMomentum`: Momentum via random coordinates. -->
* **Projections (`torchzero/modules/projections/`)**: Gradient projection techniques.
  * `FFTProjection`, `DCTProjection`: Use any update rule in Fourier or DCT domain.
  * `VectorProjection`, `TensorizeProjection`, `BlockPartition`, `TensorNormsProjection`: Structural projection methods.
  <!-- * *(Note: DCT and Galore were commented out in the `__init__.py` I read, might be experimental or moved).* -->
* **Quasi-Newton (`torchzero/modules/quasi_newton/`)**: Approximate second-order optimization methods.
  * `LBFGS`: Limited-memory BFGS.
  * `LSR1`: Limited-memory SR1.
  * `OnlineLBFGS`: Online LBFGS.
  <!-- * `ModularLBFGS`: A modular L-BFGS implementation (from experimental). -->
  * `BFGS`, `SR1`, `DFP`, `BroydenGood`, `BroydenBad`, `Greenstadt1`, `Greenstadt2`, `ColumnUpdatingMethod`, `ThomasOptimalMethod`, `PSB`, `Pearson2`, `SSVM`: Classic full-matrix Quasi-Newton update formulas.
  * Conjugate Gradient methods: `PolakRibiere`, `FletcherReeves`, `HestenesStiefel`, `DaiYuan`, `LiuStorey`, `ConjugateDescent`, `HagerZhang`, `HybridHS_DY`.
* **Second Order (`torchzero/modules/second_order/`)**: Second order methods.
  * `Newton`: Classic Newton's method.
  * `NewtonCG`: Matrix-free newton's method with conjugate gradient solver.
  * `NystromSketchAndSolve`: Nyström sketch-and-solve method.
  * `NystromPCG`: NewtonCG with Nyström preconditioning.
* **Smoothing (`torchzero/modules/smoothing/`)**: Techniques for smoothing the loss landscape or gradients.
  * `LaplacianSmoothing`: Laplacian smoothing for gradients.
  * `GaussianHomotopy`: Smoothing via randomized Gaussian homotopy.
* **Weight Decay (`torchzero/modules/weight_decay/`)**: Weight decay implementations.
  * `WeightDecay`: Standard L2 or L1 weight decay.
  <!-- * `DirectWeightDecay`: Applies weight decay directly to weights.
  * `decay_weights_`: Functional form for decaying weights. -->
* **Ops (`torchzero/modules/ops/`)**: Various tensor operations and utilities.
  * `GradientAccumulation`: easy way to add gradient accumulation.
  * `Unary*` (e.g., `Abs`, `Sqrt`, `Sign`): Unary operations.
  * `Binary*` (e.g., `Add`, `Mul`, `Graft`): Binary operations.
  * `Multi*` (e.g., `ClipModules`, `LerpModules`): Operations on multiple module outputs.
  * `Reduce*` (e.g., `Mean`, `Sum`, `WeightedMean`): Reduction operations on multiple module outputs.

* **Wrappers (`torchzero/modules/wrappers/`)**.
  * `Wrap`: Wraps any PyTorch optimizer, allowing to use it as a module.

<!-- * **Experimental (`torchzero/modules/experimental/`)**: Experimental modules.
  * `GradMin`: Attempts to minimize gradient norm.
  * `ReduceOutwardLR`: Reduces learning rate for parameters with outward pointing gradients.
  * `RandomSubspacePreconditioning`, `HistorySubspacePreconditioning`: Preconditioning techniques using random or historical subspaces. -->

## Advanced Usage

### Closure

Certain modules, particularly line searches and gradient approximations require a closure, similar to L-BFGS in PyTorch. In TorchZero closure accepts an additional `backward` argument, refer to example below:

```python
# basic training loop
for inputs, targets in dataloader:

    def closure(backward=True): # make sure it is True by default
        preds = model(inputs)
        loss = criterion(preds, targets)

        if backward:
            optimizer.zero_grad()
            loss.backward()

        return loss

    loss = optimizer.step(closure)
```

Also the closure above works with all PyTorch optimizers and most custom ones, so there is no need to rewrite the training loop.

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

opt = tz.Modular([W], tz.m.NewtonCG(), tz.m.StrongWolfe())
for step in range(20):
    loss = opt.step(closure)
    print(f'{step} - {loss}')
```

### Low level modules

TorchZero provides a lot of low-level modules that can be used to recreate update rules, or combine existing update rules
in new ways. Here are some equivalent ways to make Adam in order of their involvement:

```python
tz.m.Adam()
```

```python
tz.m.RMSprop(0.999, debiased=True, init='zeros', inner=tz.m.EMA(0.9))
```

```python
tz.m.DivModules(
    tz.m.EMA(0.9, debiased=True),
    [tz.m.SqrtEMASquared(0.999, debiased=True, amsgrad=amsgrad), tz.m.Add(1e-8)]
)
```

```python
tz.m.DivModules(
    [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9, beta2=0.999)],
    [tz.m.EMASquared(0.999, amsgrad=amsgrad), tz.m.Sqrt(), tz.m.Add(1e-8)]
)
```

```python
tz.m.DivModules(
    [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9)],
    [
        tz.m.Pow(2),
        tz.m.EMA(0.999),
        tz.m.AccumulateMaximum() if amsgrad else tz.m.Identity(),
        tz.m.Sqrt(),
        tz.m.Debias2(beta=0.999),
        tz.m.Add(1e-8)]
)
```

There are practically no rules to the ordering of the modules - anything will work, even line search after line search or nested gaussian homotopy.

### Quick guide to implementing new modules

Modules are quite similar to torch.optim.Optimizer, the main difference is that everything is stored in the Vars object,
not in the module itself. Also both per-parameter settings and state are stored in per-parameter dictionaries. Feel free to modify the example below.

```python
import torch
from torchzero.core import Module, Vars

class HeavyBall(Module):
    def __init__(self, momentum: float = 0.9, dampening: float = 0):
        defaults = dict(momentum=momentum, dampening=dampening)
        super().__init__(defaults)

    def step(self, vars: Vars):
        # a module takes a Vars object, modifies it or creates a new one, and returns it
        # Vars has a bunch of attributes, including parameters, gradients, update, closure, loss
        # for now we are only interested in update, and we will apply the heavyball rule to it.

        params = vars.params
        update = vars.get_update() # list of tensors

        exp_avg_list = []
        for p, u in zip(params, update):
            state = self.state[p]
            settings = self.settings[p]
            momentum = settings['momentum']
            dampening = settings['dampening']

            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p)

            buf = state['momentum_buffer']
            u *= 1 - dampening

            buf.mul_(momentum).add_(u)

            # clone because further modules might modify exp_avg in-place
            # and it is part of self.state
            exp_avg_list.append(buf.clone())

        # set new update to vars
        vars.update = exp_avg_list
        return vars
```

There are a some specialized base modules.

* `GradApproximator` for gradient approximations
* `LineSearch` for line searches
* `Preconditioner` for gradient preconditioners
* `QuasiNewtonH` for full-matrix quasi-newton methods that update hessian inverse approximation (because they are all very similar)
* `ConguateGradientBase` for conjugate gradient methods, basically the only difference is how beta is calculated.

## License

This project is licensed under the MIT License

## Project Links

TODO (there are docs but from very old version)
<!-- * **Homepage**: `https://torchzero.github.io/torchzero/` (Placeholder - update if available)
* **Repository**: `https://github.com/torchzero/torchzero` (Assuming this is the correct path) -->

## Other stuff

There are also wrappers providing `torch.optim.Optimizer` interface for for `scipy.optimize`, NLOpt and Nevergrad.

They are in `torchzero.optim.wrappers.scipy.ScipyMinimize`, `torchzero.optim.wrappers.nlopt.NLOptOptimizer`, and `torchzero.optim.wrappers.nevergrad.NevergradOptimizer`. Make sure closure has `backward` argument as described in **Advanced Usage**.
