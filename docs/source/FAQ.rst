FAQ
###########

How to construct modular optimizers?
=====================================
A modular optimizer can be created using the :py:class:`tz.Modular<torchzero.optim.Modular>` class. It can be constructed as :code:`tz.Modular(params, *modules)`, or as :code:`tz.Modular(params, [modules])`.

All modules are available in :py:mod:`tz.m<torchzero.modular>` namespace, e.g. :py:class:`tz.m.Adam<torchzero.modules.Adam>`.

.. code:: python

    import torchzero as tz
    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.Cautious(),
        tz.m.LR(1e-3),
        tz.m.WeightDecay(),
    )

In this examples, we're constructing an optimizer with four modules:

* Adam: applies the Adam update rule to the gradients an passes to the :code:`Cautious` module.
* Cautious: Applies "cautioning" to the update and passes it to :code:`LR`.
* LR(1e-3): Scales the update by 1e-3 and passes it to :code:`WeightDecay`.
* WeightDecay: Adds a weight decay penalty to the update. Since this is applied after :code:`Adam` and :code:`LR`, weight decay is fully decoupled.

The resulting update is subtracted from model parameters.

It is recommended to always add an :py:class:`tz.m.LR<torchzero.modules.LR>` module to support lr schedulers and per-layer learning rates (see :ref:`how do we handle learning rates?`).

Most modules perform gradient transformations, so they take in an ascent direction, which is initially the gradient, transform it in some way, and pass to the next module. The first module in the chain usually uses the gradient as the initial ascent direction.

Certain modules, such as gradient-approximation ones or :py:class:`tz.m.ExactNewton<torchzero.modules.ExactNewton>`, create an ascent direction "from scratch", so they should be placed first in the chain.

Any external PyTorch optimizer can also be used as a chainable module by using :py:class:`tz.m.Wrap<torchzero.modules.Wrap>` and :py:class:`tz.m.WrapClosure<torchzero.modules.WrapClosure>` (see :ref:`How to use external PyTorch optimizers as chainable modules?`).

A list of all modules is available at https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html

How to perform optimization?
============================
Using torchzero optimizers is generally similar to using built-in PyTorch optimizers. Here's a typical training loop:

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


Some modules and optimizers in torchzero, particularly line-search methods and gradient approximation modules, require a closure function. This is similar to how :code:`torch.optim.LBFGS` works in PyTorch. In torchzero, closure needs to accept a boolean backward argument (though the argument can have any name). When :code:`backward=True`, the closure should zero out gradients using :code:`opt.zero_grad()`, and compute gradients using :code:`loss.backward()`.

Here's how a training loop with a closure looks:

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

Note that all built-in pytorch optimizers, as well as most custom ones, support closure too! So the code above will work with all other optimizers out of the box, and you can switch between different optimizers without rewriting your training loop.

If you intend to use gradient-free methods, :code:`backward` argument is still required in the closure. Simply leave it unused. Gradient-free and gradient approximation methods always call closure with :code:`backward=False`.

How to use learning rate schedulers?
=============================================
There are two primary methods for using learning rate schedulers.

You can directly pass a learning rate scheduler class or constructor to the scheduler_cls argument of the :py:class:`tz.m.LR<torchzero.modules.LR>` module:

.. code:: python

    from torch.optim.lr_scheduler import OneCycleLR

    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.LR(1e-1, scheduler_cls = lambda opt: OneCycleLR(opt, max_lr = 1e-1, total_steps = 60_000)),
        tz.m.WeightDecay(),
    )

This method has the advantage of supporting momentum cycling. Some schedulers, like :code:`OneCycleLR`, not only adjust the learning rate but also cycle momentum parameters. When using scheduler_cls with tz.m.LR, momentum cycling will be automatically applied to all modules in your optimizer chain that have :code:`momentum` or :code:`beta1` parameters.

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
PyTorch allows you to set different options, such as learning rates, for different layers or parameter groups using parameter groups. `torchzero` offers a similar mechanism for modular optimizers.

You can define parameter groups as a list of dictionaries, just like in PyTorch. Each dictionary specifies the parameters and any custom settings for that group.

.. code:: python

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-2, 'eps': 1e-5},
        {'params': model.decoder.parameters()}
    ]

    optimizer = tz.Modular(
        param_groups,
        [tz.m.Adam(), tz.m.LR(1e-3), tz.m.WeightDecay()]
    )

In this example:

* Parameters in :code:`model.encoder` will use a learning rate of 1e-2 and a custom Adam eps value of 1e-5.
* Parameters in :code:`model.decoder` will use the default learning rate of 1e-3 and the default eps value.

Important Catch: Setting Scope
+++++++++++++++++++++++++++++++
When you specify a parameter like eps in the parameter groups, it will be applied to all modules in your optimizer chain that have an eps parameter. This can sometimes lead to unintended side effects.

For instance, both :py:class:`tz.m.Adam<torchzero.modules.Adam>` and :py:class:`tz.m.RandomizedFDM<torchzero.modules.RandomizedFDM>` modules have an eps parameter, but they have completely different meanings and value ranges in each module. Applying an eps setting intended for Adam to RandomizedFDM could cause unexpected behavior.

To avoid this issue and ensure settings are applied to the intended modules, use the :code:`set_params` method. This allows you to pass parameter groups specifically to a particular module.

.. code:: python

    adam_param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-2, 'eps': 1e-5},
        {'params': model.decoder.parameters()}
    ]

    # 1. Create the Adam module
    adam = tz.m.Adam()

    # 2. Apply custom parameter groups to the Adam module using set_params
    adam.set_params(adam_param_groups)

    # 3. Create the modular optimizer, passing the configured Adam module
    optimizer = tz.Modular(
        model.parameters(),
        [adam, tz.m.LR(1e-3), tz.m.WeightDecay()]
    )


You don't have to worry about this if you are only setting per-layer lr, because the only module that has an :code:`lr` setting is :py:class:`tz.m.LR<torchzero.modules.LR>` (see :ref:`How do we handle learning rates?`).

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

Here is an example of converting :code:`LaProp` optimizer from `pytorch_optimizer <https://pytorch-optimizers.readthedocs.io/en/latest/optimizer/#pytorch_optimizer.LaProp>`_ library into a module and chain it with :py:class:`tz.m.Cautious<torchzero.modules.Cautious>`

.. code:: py

    from pytorch_optimizer import LaProp

    tz.Modular(
        model.parameters(),
        tz.m.ClipNorm(1),
        tz.m.Wrap(LaProp, lr = 1, betas = (0.9, 0.99)),
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
Please refer to pytorch docs https://pytorch.org/tutorials/beginner/saving_loading_models.html.

Like pytorch optimizers, torchzero modular optimizers and modules support :code:`opt.state_dict()` and :code:`opt.load_state_dict()`, which saves and loads state dicts of all modules, including nested ones.

So you can use the standard code for saving and loading:

.. code:: python

    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                ...
                }, PATH)

    model = TheModelClass(*args, **kwargs)
    optimizer = tz.Modular(model.parameters(), *modules)

    checkpoint = torch.load(PATH, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


How much overhead does a torchzero modular optimizer have compared to a normal optimizer?
==========================================================================================
Since some optimizers, like Adam, have learning rate baked into the update rule, but we use LR module instead, that requires an extra add operation. Currently if :code:`tz.m.Adam` or :code:`tz.m.Wrap` are directly followed by a :code:`tz.m.LR`, they will be automatically fused (:code:`Wrap` fuses only when wrapped optimizer has an :code:`lr` parameter) to mitigate that. However adding LR fusing to all modules with a learning rate is not a priority. From what I can tell this overhead is negligible.

Whenever possible I used `_foreach_xxx <https://pytorch.org/docs/stable/torch.html#foreach-operations>`_ operations. Those operate on all parameters at once instead of using a slow python for-loops. This makes the optimizers way quicker, especially with a lot of different parameter tensors. Also all modules change the update in-place whenever possible.

Is there support for complex-valued parameters?
=================================================
Pass :code:`[i.view_as_real() for i in model.parameters()]` as parameters.

Is there support for optimized parameters being on different devices?
======================================================================
Maybe, I need to test this.

Is there support for FSDP (FullyShardedDataParallel)?
======================================================
There is no support for FDSP. It may be possible to add some FDSP module, I will look into it at some point. Currently I don't think I can even use FDSP because I only have one laptop.

Is there support for differentiable optimizers?
======================================================
There is no support for differentiable optimizers.

In PyTorch most optimizers have a :code:`differentiable` argument runs autograd through optimizer step, for example :code:`torch.optim.Adam(params, 1e-3, differentiable=True)`.

I have not looked into this yet, adding support may or may not be as easy as switching :code:`@torch.no_grad` decorator to :code:`@_use_grad_for_differentiable`.