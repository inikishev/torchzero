FAQ
###########

How to construct modular optimizers?
=====================================
A modular optimizer can be created using the :py:class:`tz.m.Modular<torchzero.optim.Modular>` class. It can be constructed as :code:`tz.Modular(params, *modules)`, or as :code:`tz.Modular(params, [modules])`.

All modules are available in :py:mod:`tz.m<torchzero.modular>` namespace, e.g. :py:class:`tz.m.Adam<torchzero.modules.Adam>`.

.. code:: python

    import torchzero as tz

    # construct it like this
    opt = tz.Modular(
        model.parameters(),
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.Cautious(), tz.m.WeightDecay()]
    )

    # or like this
    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.LR(1e-3),
        tz.m.Cautious(),
        tz.m.WeightDecay(),
    )

In the example above, :code:`Adam`, being the first module, takes in the gradient, applies the adam update rule, and passes the resulting update the next next module - :code:`LR`. It multiplies the update by the learning rate and passes it to :code:`Cautious`, which applies cautioning and passes it to :code:`WeightDecay`, which adds a weight decay penalty. The resulting update is then subtracted from the model parameters.

It is recommended to always add an :py:class:`tz.m.LR<torchzero.modules.LR>` module to support lr schedulers and per-layer learning rates (see :ref:`how do we handle learning rates?`).

Most modules perform gradient transformations, so they take in an ascent direction, which is initially the gradient, transform it in some way, and pass to the next module. The first module in the chain usually uses the gradient as the initial ascent direction.

Certain modules, such as gradient-approximation ones or :py:class:`tz.m.ExactNewton<torchzero.modules.ExactNewton>`, create an ascent direction "from scratch", so they should be placed first in the chain.

Any external PyTorch optimizer can also be used as a chainable module by using :py:class:`tz.m.Wrap<torchzero.modules.Wrap>` and :py:class:`tz.m.WrapClosure<torchzero.modules.WrapClosure>` (see :ref:`How to use external PyTorch optimizers as chainable modules?`).


How to perform optimization?
============================
Most torchzero optimizers can be used in the same way as built in pytorch optimizers:

.. code:: python

    import torchzero as tz

    opt = tz.Modular(
        model.parameters(),
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

    for inputs, targets in dataloader:
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        opt.zero_grad()


A few modules and optimizers require closure, similar to :code:`torch.optim.LBFGS` but with an additional :code:`backward` argument, which, if True, calls :code:`opt.zero_grad()` and :code:`loss.backward()`. The name of the argument doesn't matter, but I will refer to it as :code:`backward`.

All line-searches and gradient approximation modules, as well as a few other ones, require a closure. Training loop with a closure looks like this:

.. code:: python

    import torchzero as tz

    opt = tz.Modular(
        model.parameters(),
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

    for inputs, targets in dataloader:

        def closure(backward=True):
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            if backward:
                opt.zero_grad()
                loss.backward()
            return loss

        loss = opt.step(closure)

Note that all built-in pytorch optimizers, as well as most custom ones, support closure too! So the training loop above will work with all other optimizers out of the box, and switching to it prevents having to rewrite training loop when changing optimizers.

If you are intending to use gradient-free methods, :code:`backward` argument is still required in the closure. Simply leave it unused. Gradient-free and gradient approximation methods always call closure with :code:`backward=False`.

How to use learning rate schedulers?
=============================================
There are two primary methods for using learning rate schedulers.
One method is to pass learning rate scheduler class to the :py:class:`tz.m.LR<torchzero.modules.LR>` module like this:

.. code:: python

    from torch.optim.lr_scheduler import OneCycleLR

    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.LR(1e-1, scheduler_cls = lambda opt: OneCycleLR(opt, max_lr = 1e-1, total_steps = 60_000)),
        tz.m.WeightDecay(),
    )

This method also supports cycling momentum, which some schedulers like OneCycleLR do. Momentum will be cycled on all modules that have :code:`momentum` or :code:`beta1` parameters.

Alternatively, learning rate scheduler can be created separately by passing it the LR module, which can be accessed with :py:meth:`get_lr_module<torchzero.optim.Modular.get_lr_module>` method like this:

.. code:: python

    opt = tz.Modular(
        model.parameters(),
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

    scheduler = OneCycleLR(opt.get_lr_module(), max_lr = 1e-1, total_steps=60_000)

Here :code:`get_lr_module` returns the :py:class:`tz.m.LR<torchzero.modules.LR>`, even if it is nested somewhere. You can then call :code:`scheduler.step()` as usual. This method does not support cycling momentum.


How to specify per-parameter options?
=============================================
In pytorch it is possible to specify per-layer options, such as learning rate, using parameter groups. In torchzero those are specified in almost the same way (although there is a catch):

.. code:: python

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-2, 'eps': 1e-5},
        {'params': model.decoder.parameters()}
    ]

    optimizer = tz.Modular(
        param_groups,
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

In the example above, :code:`model.encoder` will use a custom learning rate of 1e-2, and custom adam epsilon of 1e-5, while :code:`model.decoder` will stick to the default learning rate of 1e-3 and the default epsilon value.

The catch is that when you specify a setting such as `eps`, it will be applied to ALL modules that have that setting, which may lead to unexpected behavior. For example, both :py:class:`tz.m.Adam<torchzero.modules.Adam>` and :py:class:`tz.m.RandomizedFDM<torchzero.modules.RandomizedFDM>` have an `eps` parameter, which has completely different function and value range. To avoid this, per-parameter settings can be specified for specific modules by using the `set_params` method:

.. code:: python

    adam_param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-2, 'eps': 1e-5},
        {'params': model.decoder.parameters()}
    ]

    # 1. create adam
    adam = tz.m.Adam()

    # 2. pass custom parameter groups to adam
    adam.set_params(adam_param_groups)

    # 3. create modular optimizer after passing custom parameter groups,
    # pass it normal model.parameters()
    optimizer = tz.Modular(
        model.parameters(),
        [adam, tz.m.LR(1e-3), tz.m.WeightDecay()]
    )


You don't have to worry about this if you are only setting per-layer lr, because the only module that has an :code:`lr` setting is :py:class:`tz.m.LR` (see :ref:`How do we handle learning rates?`).

How do we handle learning rates?
=================================
Certain optimisers, like Adam, have learning rate built into the update rule. Using multiple such modules can result in unintended compounding of learning rate modifications.

To avoid this, learning rate should be applied by a singular :py:class:`tz.m.LR<torchzero.modules.LR>` module. All other modules with a learning rate, such as :py:class:`tz.m.Adam<torchzero.modules.Adam>`, have `lr` renamed to `alpha` with the default value of 1 to avoid rescaling the update.

For example:

.. code:: python

    tz.Modular(
        model.parameters(),
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

Here, instead of using Adam's `alpha` setting, we added an :code:`LR` module. This allows this modular optimizer to support per-parameter `lr` setting and learning rate schedulers, without having to worry about learning rate compounding.

See also:

* :ref:`how to use learning rate schedulers?`
* :ref:`How to specify per-parameter options?`

How to use external PyTorch optimizers as chainable modules?
============================================================
In addition to torchzero modules, any PyTorch optimizer can be used as a module using :py:class:`tz.m.Wrap<torchzero.modules.Wrap>`.

There are two slightly different ways to construct a :code:`Wrap` module. Here I will convert :code:`LaProp` optimizer from `pytorch_optimizer <https://pytorch-optimizers.readthedocs.io/en/latest/optimizer/#pytorch_optimizer.LaProp>`_ library into a module and chain it with :py:class:`tz.m.Cautious<torchzero.modules.Cautious>`

.. code:: py

    from pytorch_optimizer import LaProp

    # first way
    tz.Modular(
        model.parameters(),
        tz.m.ClipNorm(1),
        tz.m.Wrap(LaProp, lr = 1, betas = (0.9, 0.99)),
        tz.m.LR(1e-3),
        tz.m.Cautious(),
    )

    # second way (identical but more verbose)
    tz.Modular(
        model.parameters(),
        tz.m.ClipNorm(1),
        tz.m.Wrap(LaProp(model.parameters(), lr = 1, betas = (0.9, 0.99))),
        tz.m.LR(1e-3),
        tz.m.Cautious(),
    )

Most pytorch optimizers update model parameters by using their :code:`.grad` attibute. Wrap puts the current update into the :code:`.grad`, making the wrapped optimizer use it instead.

Note that since the wrapped optimizer updates model parameters directly, if :class:`Wrap` is not the last module, it stores model parameters before the step, then performs a step with the wrapped optimizer, calculates the update as difference between model parameters before and after the step, undoes the step, and passes the update to the next module. That may introduce additional overhead compared to using modules.

However when :py:class:`Wrap` is the last module in the chain, it simply makes a step with the wrapped optimizer, so no overhead is introduced.

Also notice how I set `lr` to 1 in LaProp, and instead used an :py:class:`tz.m.LR<torchzero.modules.LR>` module. As usual, to make the optimizer support lr scheduling and per-layer learning rates, use the :py:class:`LR` module to set the learning rate. Alternatively pass per-layer parameters or apply scheduling directly to LaProp optimizer, before wrapping it.

There is also a :py:class:`tz.m.WrapClosure<torczhero.modules.WrapClosure>` for optimizers that require closure, such as :code:`torch.optim.LBFGS`. It modifies the closure to set :code:`.grad` attribute on each closure evaluation. So you can use LBFGS with FDM or gradient smoothing methods.

How to save/serialize a modular optimizer?
============================================
TODO

How much overhead does a torchzero modular optimizer have compared to a normal optimizer?
==========================================================================================
A thorough benchmark will be posted to this section very soon. There is no overhead other than what is described below.

Since some optimizers, like Adam, have learning rate baked into the update rule, but we use LR module instead, that requires an extra add operation. Currently if :code:`tz.m.Adam` or :code:`tz.m.Wrap` are directly followed by a :code:`tz.m.LR`, they will be automatically fused (:code:`Wrap` fuses only when wrapped optimizer has an :code:`lr` parameter). However adding LR fusing to all modules with a learning rate is not a priority.

Whenever possible I used `_foreach_xxx <https://pytorch.org/docs/stable/torch.html#foreach-operations>`_ operations. Those operate on all parameters at once instead of using a slow python for-loops. This makes the optimizers way quicker, especially with a lot of different parameter tensors. Also all modules change the update in-place whenever possible.

Is there support for complex-valued parameters?
=================================================
Currently no, as I have not made the modules with complex-valued parameters in mind, although some might still work. I do use complex-valued networks so I am looking into adding support. There may actually be a way to support them automatically.

Is there support for optimized parameters being on different devices?
======================================================================
TODO

Is there support for FSDP (FullyShardedDataParallel)?
======================================================
There is no support for FDSP. It may be possible to add some FDSP module, I will look into it at some point. Currently I don't think I can even use FDSP because I only have one laptop.

Is there support for differentiable optimizers?
======================================================
There is no support for differentiable optimizers.

In PyTorch most optimizers have a :code:`differentiable` argument runs autograd through optimizer step, for example :code:`torch.optim.Adam(params, 1e-3, differentiable=True)`.

I have not looked into this yet, adding support may or may not be as easy as switching :code:`@torch.no_grad` decorator to :code:`@_use_grad_for_differentiable`.