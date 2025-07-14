![tests](https://github.com/inikishev/torchzero/actions/workflows/tests.yml/badge.svg)

# torchzero

**Modular optimization library for PyTorch**

`torchzero` is a PyTorch library providing a highly modular framework for creating and experimenting with a huge number of various optimization algorithms - various momentum techniques, gradient clipping, gradient approximations, line searches, quasi newton methods and more. All algorithms are implemented as modules that can be chained together freely.

## Installation

```bash
pip install torchzero
```

pip version is always the latest one. Or install from this repo

```bash
pip install git+https://github.com/inikishev/torchzero
```

**Dependencies:**

* Python >= 3.10
* `torch`
* `numpy`
* `typing_extensions`

## Quick Start / Usage Example

Basic example:

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
# 1. ClipValue: clips gradients to (-10, 10) range.
# 2. Adam: applies Adam update rule to clipped gradients.
# 3. NormalizeByEMA: stabilizes the update by normalizing it to an exponential
# moving average of past updates.
# 4. WeightDecay - decoupled weight decay (can also move after LR to fully decouple)
# 5. LR: Scales the computed update by the learning rate (supports LR schedulers).
optimizer = tz.Modular(
    model.parameters(),
    tz.m.ClipValue(10),
    tz.m.Adam(),
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

`torchzero` provides a huge number of various modules:

* **Optimizers**: Optimization algorithms.
  * `Adam`, `Adan`, `Adagrad`, `ESGD`, `FullMatrixAdagrad`, `LMAdagrad`, `AdaHessian`, `AdaptiveHeavyBall`, `OrthoGrad`, `Lion`, `MARS`, `MatrixMomentum`, `AdaptiveMatrixMomentum`, `Muon`, `RMSprop`, `Rprop`, `SAM`, `ASAM`, `MSAM`, `Shampoo`, `SOAP`, `SophiaH`.

  Additionally many other optimizers can be easily defined via modules:
  * Grams: `[tz.m.Adam(), tz.m.GradSign()]`
  * LaProp: `[tz.m.RMSprop(), tz.m.EMA(0.9)]`
  * Signum: `[tz.m.HeavyBall(), tz.m.Sign()]`
  * Efficient full-matrix version of any diagonal optimizer, like Adam: `[tz.m.LMAdagrad(beta=0.999, inner=tz.m.EMA(0.9)), tz.m.Debias(0.9, 0.999)]`
  * Cautious version of any optimizer, like SOAP: `[tz.m.SOAP(), tz.m.Cautious()]`

* **Momentum**:
  * `HeavyBall`: Classic momentum (Polyak's momentum).
  * `NAG`: Nesterov Accelerated Gradient.
  * `EMA`: Exponential moving average.
  * `Averaging` (`MedianAveraging`, `WeightedAveraging`): Simple, median, or weighted averaging of updates.
  * `Cautious`, `ScaleByGradCosineSimilarity`: Momentum cautioning.

* **Stabilization**: Gradient stabilization techniques.
  * `ClipNorm`: Clips gradient L2 norm.
  * `ClipValue`: Clips gradient values element-wise.
  * `Normalize`: Normalizes gradients to unit norm.
  * `Centralize`: Centralizes gradients by subtracting the mean.
  * `ClipNormByEMA`, `NormalizeByEMA`, `ClipValueByEMA`: Clipping/Normalization based on EMA of past values.
  * `ClipNormGrowth`, `ClipValueGrowth`: Limits norm or value growth.

* **Gradient approximations**: Methods for approximating gradients.
  * `FDM`: Finite difference method.
  * `RandomizedFDM` (`MeZO`, `SPSA`, `RDSA`, `Gaussian smoothing`): Randomized finite difference methods (also subspaces).
  * `ForwardGradient`: Randomized gradient approximation via forward mode automatic differentiation.

* **Second order**: Second order methods.
  * `Newton`: Classic Newton's method.
  * `InverseFreeNewton`: Inverse-free version of Newton's method.
  * `NewtonCG`: Matrix-free newton's method with conjugate gradient or minimal residual solvers.
  * `TruncatedNewtonCG`: Steihaug-Toint Trust-region NewtonCG via a truncated CG solver.
  * `NystromSketchAndSolve`: Nyström sketch-and-solve method.
  * `NystromPCG`: NewtonCG with Nyström preconditioning.
  * `HigherOrderNewton`: Higher order Newton's method with trust region.

* **Quasi-Newton**: Approximate second-order optimization methods.
  * `LBFGS`: Limited-memory BFGS.
  * `LSR1`: Limited-memory SR1.
  * `OnlineLBFGS`: Online LBFGS.
  * `BFGS`, `DFP`, `ICUM`, `PSB`, `SR1`, `SSVM`, `BroydenBad`, `BroydenGood`, `FletcherVMM`, `GradientCorrection`, `Greenstadt1`, `Greenstadt2`, `Horisho`, `McCormick`, `NewSSM`, `Pearson`, `ProjectedNewtonRaphson`, `ThomasOptimalMethod`, `ShorR`: Full-matrix quasi-newton methods.
  * `DiagonalBFGS`, `DiagonalSR1`, `DiagonalQuasiCauchi`, `DiagonalWeightedQuasiCauchi`, `DNRTR`, `NewDQN`: Diagonal quasi-newton methods.
  * `PolakRibiere`, `FletcherReeves`, `HestenesStiefel`, `DaiYuan`, `LiuStorey`, `ConjugateDescent`, `HagerZhang`, `HybridHS_DY`, `ProjectedGradientMethod`: Conjugate gradient methods.

* **Trust Region** Trust region can work with exact hessian or any of the quasi-newton methods (L-BFGS support is WIP)
  * `TrustCG`: Trust-region, uses a Steihaug-Toint truncated CG solver.
  * `CubicRegularization`: Cubic regularization, works better with exact hessian.

* **Line Search**:
  * `Backtracking`, `AdaptiveBacktracking`: Backtracking line searches (adaptive is my own).
  * `StrongWolfe`: Cubic interpolation line search satisfying strong Wolfe conditions.
  * `ScipyMinimizeScalar`: Wrapper for SciPy's scalar minimization for line search.

* **Learning Rate**:
  * `LR`: Controls learning rate and adds support for LR schedulers.
  * `PolyakStepSize`: Polyak's subgradient method.
  * `BarzilaiBorwein`: Barzilai-Borwein step-size.
  * `Warmup`, `WarmupNormCLip`: Learning rate warmup.

* **Projections**: This can implement things like GaLore but I haven't done that yet.
  <!-- * `FFTProjection`, `DCTProjection`: Use any update rule in Fourier or DCT domain (doesn't seem to help though).
  * `VectorProjection`, `TensorizeProjection`, `BlockPartition`, `TensorNormsProjection`: Structural projection methods (for block BFGS etc.). -->
  This is WIP
  * `To`: this casts everything to any other dtype and device for other modules, e.g. if you want better precision
  * `ViewAsReal`: put if you have complex paramters.

* **Smoothing**: Smoothing-based optimization methods.
  * `LaplacianSmoothing`: Laplacian smoothing for gradients (implements Laplacian Smooth GD).
  * `GaussianHomotopy`: Smoothing via randomized Gaussian homotopy.

* **Weight Decay**:.
  * `WeightDecay`: Standard L2 or L1 weight decay.

* **Ops**: This has low level operations, also stuff like grafting and gradient accumulation.

* **Wrappers**.
  * `Wrap`: Wraps any PyTorch optimizer, allowing to use it as a module.

* **Experimental**: various horrible atrocities

A complete list of modules is available in the [documentation](https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html).

## Advanced Usage

### Closure

Certain modules, particularly line searches and gradient approximations require a closure, similar to L-BFGS in PyTorch. Also some modules require closure to accept an additional `backward` argument, refer to example below:

```python
# basic training loop
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

opt = tz.Modular([W], tz.m.NewtonCG(), tz.m.StrongWolfe())
for step in range(20):
    loss = opt.step(closure)
    print(f'{step} - {loss}')
```

### Module combinations

There are practically no rules to the ordering of the modules - anything will work. For example any method can be made zeroth order by putting it after some gradient approximation module such as GaussianSmoothing:

```python
opt = tz.Modular(
    bench.parameters(),
    tz.m.GaussianSmoothing(h=0.01, n_samples=10),
    tz.m.NewtonCG(hvp_method='forward'),
    tz.m.AdaptiveBacktracking(),
)
```

GaussianSmoothing actually creates a new **closure** which approximates the gradient. To NewtonCG this closure is just like
any other closure, so it works seamlessly.

Any module can be projected (this is how it will work once I implement GaLore, but I haven't done that yet):

```python
tz.m.GaLore([tz.m.GraftModules(tz.m.Shampoo(), tz.m.RMSprop()), tz.m.LR(1e-2)])
```

### Low level modules

torchzero provides a lot of low-level modules that can be used to recreate update rules, or combine existing update rules
in new ways. Here are some equivalent ways to make Adam in order of their involvement:

```python
tz.m.Adam()
```

```python
# Adam is debiased RMSprop applied to EMA
tz.m.RMSprop(0.999, debiased=True, init='zeros', inner=tz.m.EMA(0.9))
```

```python
tz.m.DivModules(
    tz.m.EMA(0.9, debiased=True),
    [tz.m.SqrtEMASquared(0.999, debiased=True), tz.m.Add(1e-8)]
)
```

```python
tz.m.DivModules(
    [tz.m.EMA(0.9), tz.m.Debias(beta1=0.9, beta2=0.999)],
    [tz.m.EMASquared(0.999), tz.m.Sqrt(), tz.m.Add(1e-8)]
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

### Quick guide to implementing new modules

Modules are quite similar to torch.optim.Optimizer, the main difference is that everything is stored in the Vars object,
not in the module itself. Also both per-parameter settings and state are stored in per-parameter dictionaries. Feel free to modify the example below.

```python
import torch
from torchzero.core import Module, Var

class HeavyBall(Module):
    def __init__(self, momentum: float = 0.9, dampening: float = 0):
        defaults = dict(momentum=momentum, dampening=dampening)
        super().__init__(defaults)

    def step(self, var: Var):
        # Var object holds all attributes used for optimization - parameters, gradient, update, etc.
        # a module takes a Var object, modifies it or creates a new one, and returns it
        # Var has a bunch of attributes, including parameters, gradients, update, closure, loss
        # for now we are only interested in update, and we will apply the heavyball rule to it.

        params = var.params
        update = var.get_update() # list of tensors

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

        # set new update to var
        var.update = exp_avg_list
        return var
```

More in-depth guide will be available in the documentation in the future.

## Other stuff

There are also wrappers providing `torch.optim.Optimizer` interface for various other libraries. When using those, make sure closure has `backward` argument as described in **Advanced Usage**.

---

### Scipy

#### torchzero.optim.wrappers.scipy.ScipyMinimize

A wrapper for `scipy.optimize.minimize` with gradients and hessians supplied by pytorch autograd. Scipy provides implementations of the following methods: `'nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'`.

#### torchzero.optim.wrappers.scipy.ScipyDE, ScipyDualAnnealing, ScipySHGO, ScipyDIRECT, ScipyBrute

Equivalent wrappers for other derivative free solvers available in `scipy.optimize`

---

### NLOpt

#### torchzero.optim.wrappers.nlopt.NLOptWrapper

A wrapper for [NLOpt](https://github.com/stevengj/nlopt) with gradients supplied by pytorch autograd. NLOpt is another popular library with many gradient based and gradient free [algorithms](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/)

---

### Nevergrad

#### torchzero.optim.wrappers.nevergrad.NevergradWrapper

A wrapper for [nevergrad](https://facebookresearch.github.io/nevergrad/) which has a huge library of gradient free [algorithms](https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers)

---

### fast-cma-es

#### torchzero.optim.wrappers.fcmaes.FcmaesWrapper

A wrapper for [fast-cma-es](https://github.com/dietmarwo/fast-cma-es), which implements various gradient free algorithms. Notably it includes [BITEOPT](https://github.com/avaneev/biteopt) which seems to have very good performance in benchmarks.

# License

This project is licensed under the MIT License

# Project Links

The documentation is available at <https://torchzero.readthedocs.io/en/latest/>
