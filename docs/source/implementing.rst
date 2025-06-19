Implementing new modules
############################

Modules are quite similar to torch.optim.Optimizer, the main difference is that everything is stored in the Var object,
not in the module itself. Also both per-parameter settings and state are stored in per-parameter dictionaries. Feel free to modify the example below.

.. code:: python
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

A more in-depth guide will be written soon.

If you need more examples, check documentation on some base modules:

:py:class:`torchzero.modules.line_search.LineSearch`
:py:class:`torchzero.modules.grad_approximation.GradApproximator`
:py:class:`torchzero.modules.quasi_newton.quasi_newton.HessianUpdateStrategy`
:py:class:`torchzero.modules.quasi_newton.cg.ConguateGradientBase`
