Introduction
==================

torchzero implements a large number of chainable optimization modules that can be chained together to create custom optimizers.

Each module takes the output of the previous module and applies a further transformation. This modular design avoids redundant code, such as reimplementing cautioning, orthogonalization, laplacian smoothing, etc for every optimizer. It is also easy to experiment with grafting, interpolation between different optimizers, and perhaps some weirder combinations like nested momentum.

Modules are not limited to gradient transformations. They can perform other operations like line searches, exponential moving average (EMA) and stochastic weight averaging (SWA), gradient accumulation, gradient approximation, and more.

There are over 100 modules, all accessible within the :py:mod:`tz.m<torchzero.modular>` namespace. For example, the Adam update rule is available as :py:class:`tz.m.Adam<torchzero.modules.Adam>`. Complete list of modules is available in [documentation](https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html).

Modules can be chained with :py:class:`tz.Modular<torchzero.core.Modular>` and used as any other pytorch optimizer. Here’s an example of how to define a Cautious Adam optimizer with gradient clipping and decoupled weight decay:

.. code:: python

    import torch
    import torchzero as tz

    # define dataset and model
    inputs = torch.randn((32, 10))
    targets = torch.randint(low=0, high=2, size=(32,1)).float()

    model = torch.nn.Sequential(torch.nn.Linear(10,10), torch.nn.ReLU(), torch.nn.Linear(10, 1))
    criterion = torch.nn.BCEWithLogitsLoss()

    # define Cautious Adam with gradient clipping and decoupled weight decay
    opt = tz.Modular(
        model.parameters(),
        tz.m.ClipValue(1),
        tz.m.Adam(),
        tz.m.Cautious(),
        tz.m.WeightDecay(1e-4),
        tz.m.LR(1e-1),
    )

    # standard training loop
    for epoch in range(100):
        preds = model(inputs)
        loss = criterion(preds, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(epoch, loss.item(), end = '       \r')


Please head over to :ref:`FAQ` for more examples and information.