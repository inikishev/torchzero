# torchzero
This is a work-in-progress general purpose optimization library for pytorch. We have zeroth, first, second order and quasi newton methods, gradient approximation, line searches and a whole lot of other stuff.

Most optimizers are modular, meaning you can chain them like this:
```py
optimizer = torchzero.optim.ModularOptimizer(model.parameters(), [*list of modules*])`
```
For example you might use `[ClipNorm(4), LR(1e-3), NesterovMomentum(0.9)]` for standard SGD with gradient clipping and nesterov momentum. If you don't have access to gradients, add a `RandomizedFDM()` at the beginning to approximate them via randomized finite differences. 

Or `[ExactNewton(), BacktrackingLS()]` for newton with backtracking line search. Something a bit more interesting - `[Subspace(ProjRandom(3)), NewtonFDM(1e-3)]` will perform a newton step via finite-difference approximated hessian in a small subspace defined by 3 random projections, making it feasible for large scale problems.

# How to use

All modules are defined in `torchzero.modules`. You can generally mix and match them however you want. Some pre-made optimizers are available in `torchzero.optim`.

Many optimizers require closure, which should look like this:
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
This code will also work with all built in pytorch optimizers, including LBFGS, all optimizers in this library, as well as most custom ones.

# Stuff i've implemented
There will be docs with a more exhaustive list and explanations. For now I hope that everything should be reasonably straightforward to use.
- Gradient approximation via finite difference or randomized finite difference (which includes SPSA and Gaussian smoothing algorithm described in *Nesterov, Y., & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2), 527-566.*)
- Exact Newton's method (with Levenberg-Marquardt regularization), and newton with hessian approximation via finite difference.
- Various line searches
- Polyak momentum, nesterov momentum
- Gradient clipping and normalization
- Learning rate droput (*Lin, H., Zeng, W., Zhuang, Y., Ding, X., Huang, Y., & Paisley, J. (2022). Learning rate dropout. IEEE Transactions on Neural Networks and Learning Systems, 34(11), 9029-9039.*)
- Laplacian smoothing (*Osher, S., Wang, B., Yin, P., Luo, X., Barekat, F., Pham, M., & Lin, A. (2022). Laplacian smoothing gradient descent. Research in the Mathematical Sciences, 9(3), 55*)
- Cautious Optimizers (https://huggingface.co/papers/2411.16085)
- Projections into small random subspace (which is a part of things like *Gower, R., Kovalev, D., Lieder, F., & Richtárik, P. (2019). RSN: randomized subspace Newton. Advances in Neural Information Processing Systems, 32.*)
- I've implemented SGD and Adam as composable modules as well, so if you ever wanted Adam with line search, you can now do it (but check out *Kenneweg, P., Kenneweg, T., Fumagalli, F., & Hammer, B. (2024, June). No learning rates needed: Introducing SALSA-Stable Armijo Line Search Adaptation. In 2024 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.*)

All modules should be quite fast, especially on models with many different parameters, due to `_foreach` operations.

Also due to the modular nature of those implementations, they usually turn out to have reasonably clean code and might be good as reference implementations.

But the code is still highly experimental, untested and subject to change, so feel free but be careful if using this for actual project.


# other stuff
### scipy.optimize.minimize.wrapper
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
