FAQ
###########

How to construct modular optimizers?
=====================================
A modular optimizer can be created using the :py:class:`tz.Modular<torchzero.core.Modular>` class.

All modules are available in :py:mod:`tz.m<torchzero.modular>` namespace, e.g. :py:class:`tz.m.Adam<torchzero.modules.Adam>`.

.. code:: python

    import torchzero as tz
    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.Cautious(),
        tz.m.WeightDecay(1e-3),
        tz.m.LR(1e-3),
    )

In this examples, we're constructing an optimizer with four modules:

* Adam: applies the Adam update rule to the gradients an passes to the :code:`Cautious` module.
* Cautious: Applies "cautioning" to the update and passes it to :code:`LR`.
* WeightDecay: Adds a weight decay penalty to the update. Since this is applied after :code:`Adam`, it is decoupled.
* LR(1e-3): Scales the update by 1e-3 and passes it to :code:`WeightDecay`.

The resulting update is subtracted from model parameters.

It is recommended to always add an :py:class:`tz.m.LR<torchzero.modules.LR>` module to support lr schedulers and per-layer learning rates.

Most modules perform gradient transformations, so they take in input tensors, which is initially the gradient, transform them in some way, and pass to the next module. The first module in the chain usually uses the gradient as the initial update.

Certain modules, such as gradient-approximation ones or :py:class:`tz.m.Newton<torchzero.modules.Newton>`, create an ascent direction "from scratch", so they should be placed first in the chain.

Any external PyTorch optimizer can also be used as a chainable module by using :py:class:`tz.m.Wrap<torchzero.modules.Wrap>`.

A list of all modules is available at https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html

How to perform optimization?
============================
Using torchzero optimizers is generally similar to using built-in PyTorch optimizers. Here's a typical training loop:

.. code:: python

    import torchzero as tz

    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.WeightDecay(1e-3),
        tz.m.LR(1e-3),
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
        tz.m.Adam(),
        tz.m.WeightDecay(1e-3)
        tz.m.LR(1e-3),

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
To apply a scheduler, make sure there is a single :py:class:`tz.m.LR<torchzero.modules.LR>` module somewhere in the chain.

Then pass the optimizer to your scheduler:

.. code:: python

    import torchzero as tz
    from torch.optim.lr_scheduler import OneCycleLR

    opt = tz.Modular(
        model.parameters(),
        tz.m.Adam(),
        tz.m.WeightDecay(1e-3)
        tz.m.LR(1e-3),

    )

    scheduler = OneCycleLR(opt)


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
        tz.m.Adam(),
        tz.m.WeightDecay(1e-3)
        tz.m.LR(1e-3),
    )

In this example:

* Parameters in :code:`model.encoder` will use a learning rate of 1e-2 and a custom Adam eps value of 1e-5.
* Parameters in :code:`model.decoder` will use the default learning rate of 1e-3 and the default eps value.

Important Catch: Setting Scope
+++++++++++++++++++++++++++++++
When you specify a parameter like eps in the parameter groups, it will be applied to all modules in your optimizer chain that have an eps parameter. This can sometimes lead to unintended side effects.

For instance, both :py:class:`tz.m.Adam<torchzero.modules.Adam>` and :py:class:`tz.m.RandomizedFDM<torchzero.modules.RandomizedFDM>` modules have an eps parameter, but they have completely different meanings and value ranges in each module. Applying an eps setting intended for Adam to RandomizedFDM could cause unexpected behavior.

To avoid this issue and ensure settings are applied to the intended modules, use the :code:`set_param_groups` method. This allows you to pass parameter groups specifically to a particular module.

.. code:: python

    adam_param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-2, 'eps': 1e-5},
        {'params': model.decoder.parameters()}
    ]

    # 1. Create the Adam module
    adam = tz.m.Adam()

    # 2. Apply custom parameter groups to the Adam module using set_params
    adam.set_param_groups(adam_param_groups)

    # 3. Create the modular optimizer, passing the configured Adam module
    optimizer = tz.Modular(
        model.parameters(),
        adam,
        tz.m.WeightDecay(1e-3),
        tz.m.LR(1e-3),
    )


You don't have to worry about this if you are only setting per-layer lr, because the only module that has an :code:`lr` setting is :py:class:`tz.m.LR<torchzero.modules.LR>`.

How do we handle learning rates?
=================================
Certain optimisers, like Adam, have learning rate built into the update rule. Using multiple such modules can result in unintended compounding of learning rate modifications.

To avoid this, learning rate should be applied by a singular :py:class:`tz.m.LR<torchzero.modules.LR>` module. All other modules with a learning rate, such as :py:class:`tz.m.Adam<torchzero.modules.Adam>`, have `lr` renamed to `alpha` with the default value of 1 to avoid rescaling the update.

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

Also notice how I set `lr` to 1 in LaProp, and instead used an :py:class:`tz.m.LR<torchzero.modules.LR>` module. As usual, to make the optimizer support lr scheduling and per-layer learning rates, use the :py:class:`LR` module to set the learning rate. Alternatively pass per-layer parameters or apply scheduling directly to LaProp optimizer, before wrapping it.


How to save/serialize a modular optimizer?
============================================
Please refer to pytorch docs https://pytorch.org/tutorials/beginner/saving_loading_models.html.

Like pytorch optimizers, torchzero modular optimizers support :code:`opt.state_dict()` and :code:`opt.load_state_dict()`, which saves and loads state dicts of all modules, including nested ones.

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
