Introduction
==================

torchzero is a library for pytorch that offers a flexible and modular way to build optimizers for various tasks. By combining smaller, reusable modules, you can easily customize and experiment with different optimization strategies.

Each module takes the output of the previous module and applies a further transformation. This modular design avoids redundant code, such as reimplementing Laplacian smoothing, cautioning, orthogonalization, etc for every optimizer. It also simplifies experimenting with advanced techniques like optimizer grafting, interpolation, and complex combinations like nested momentum.

Many modules perform gradient transformations. They receive an "ascent direction," which is initially the gradient, modify it, and pass it to the next module in the chain. Typically, the first module uses the raw gradient as the starting ascent direction. However, modules are not limited to gradient transformations. They can perform other operations like line searches, exponential moving average (EMA) and stochastic weight averaging (SWA), gradient accumulation, gradient approximation, and more.

torchzero provides over 100 modules, all accessible within the :py:mod:`tz.m<torchzero.modular>` namespace. For example, the Adam module is available as :py:class:`tz.m.Adam<torchzero.modules.Adam>`. You can find a complete list of modules in the torchzero documentation: https://torchzero.readthedocs.io/en/latest/autoapi/torchzero/modules/index.html.

To combine these modules and create a custom optimizer, use tz.Modular, and then use it as any other pytorch optimizer. Here’s an example of how to define a Cautious Adam optimizer with gradient clipping and decoupled weight decay:

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
        tz.m.LR(1e-1),
        tz.m.Cautious(),
        tz.m.WeightDecay(1e-4),
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