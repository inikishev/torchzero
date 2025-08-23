# Index

All modules are awailable in ``tz.m`` namespace (e.g. ``tz.m.Adam``).
There are a lot of modules, so they are vaguely split into sub-packages, although some of them can be hard to categorize. You can also view [all modules](all.md) on a single (very long) page.

## Optimization algorithms

* [Adaptive](modules/adaptive.md) - Adaptive per-parameter learning rates + some other deep learning optimizers, e.g. Adam, etc.
* [Momentum](modules/momentum.md) - momentums and exponential moving averages.
* [Conjugate gradient](modules/conjugate_gradient.md) - conjugate gradient methods.
* [Quasi-newton](modules/quasi_newton.md) - quasi-newton methods that estimate the hessian using gradient information.
* [Second order](modules/second_order.md) - "True" second order methods that use exact second order information.
* [Higher order](modules/higher_order.md) - third and higher order methods (currently just higher order newton).
* [Gradient approximation](modules/grad_approximation.md) - modules that estimate the gradient using function values.
* [Least-squares](modules/least_squares.md) - least-squares methods (Gauss-newton)

## Step size selection

* [Step size](modules/step_size.md) - step size selection methods like Barzilai-Borwein and Polyak's step size.
* [Line search](modules/line_search.md) - line search methods.
* [Trust region](modules/trust_region.md) - trust region methods.

## Auxillary modules

* [Clipping](modules/clipping.md) - gradient clipping, normalization, centralization, etc.
* [Weight decay](modules/weight_decay.md) - weight decay.
* [Operations](modules/ops.md) - operations like adding modules, subtracting, grafting, tracking the maximum, etc.
* [Projections](modules/projections.md) - allows any other modules to be used in some projected space. This has multiple uses, one is to save memory by projecting into a smaller subspace, another is splitting parameters into smaller blocks or merging them into a single vector, another one is peforming optimization in a different dtype or viewing complex tensors as real. This can also do things like optimize in fourier domain.
* [Smoothing](modules/smoothing.md) - smoothing-based optimization, currently laplacian and gaussian smoothing are implemented.
* [Miscellaneous](modules/misc.md) - a lot of uncategorized modules, notably gradient accumulation, switching, automatic resetting, random restarts.
* [Wrappers](modules/wrappers.md) - this implements Wrap, which can turn most custom pytorch optimizers into chainable modules.


<!-- === "Gradient approximation"
    ::: torchzero.modules.grad_approximation
        options:
          show_root_heading: true
          heading_level: 3

=== "Higher-order methods"
    ::: torchzero.modules.higher_order
        options:
          show_root_heading: true
          heading_level: 3

=== "Least-squares"
    ::: torchzero.modules.least_squares
        options:
          show_root_heading: true
          heading_level: 3 -->
