======================================
Tutorial 1: Training iUNets in Pytorch
======================================

In this tutorial, we will demonstrate how to use invertible U-Net (iUNet) as
part of a model built in Pytorch. Despite having a custom backpropagation
implementation, any iUNet can be used e.g. as a submodule in a larger neural
network architecture, as well as be trained like any other neural network in
Pyroch.

An invertible toy problem
-------------------------

In the following, we will train an iUNet to *mirror* the input image as a
warm-up.

Setting up the data loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from matplotlib import pyplot as plt

    import torch
    import torchvision

    from iunets import iUNet

    # Load CIFAR10 dataset
    data = torchvision.datasets.CIFAR10(root='.',
        download=True,
        train=True, transform=
            torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
        )
    )

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=128,
        shuffle=True
    )

Defining an iUNet
~~~~~~~~~~~~~~~~~

.. code:: python

    model = iUNet(
        in_channels=3,
        dim=2,
        architecture=(4,4,4,4,4)
    ).to('cuda')

    model.print_layout()

Here, ``in_channels`` defines the number of input channels, ``dim=2`` signifies,
that we are using 2D data (images). ``architecture=(4,4,4,4,4)`` defines the
number of *resolution-preserving
The call to ``model.print_layout()`` now prints the layout of the above-defined
iUNet:

.. code:: text

    3-3-3-3-(1/2)----------------------------------------------------------------------------------------------------(1/2)-3-3-3-3
    ----------8-8-8-8-(4/4)-------------------------------------------------------------------------------(4/4)-8-8-8-8-----------
    ---------------------16-16-16-16-(8/8)--------------------------------------------------(8/8)-16-16-16-16---------------------
    -----------------------------------32-32-32-32-(16/16)-----------------(16/16)-32-32-32-32------------------------------------
    ---------------------------------------------------64-64-64-64--64-64-64-64---------------------------------------------------

Here, each number represents the number of channels. The expressions in
parentheses denote the splitting of channels, a part of which is then
invertibly downsampled (and later upsampled and re-concatenated).

Training the model
~~~~~~~~~~~~~~~~~~

Now, a training loop can be set up exactly like with any other Pytorch model:

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_curve = []
    loss_fn = torch.nn.MSELoss()

    for epoch in range(50):
        for id, (x, _) in enumerate(data_loader):
            x = x.to('cuda')
            y = torch.flip(x, (3,))
            optimizer.zero_grad()
            output = model(x)

            loss = loss_fn(output, y)
            loss.backward()
            loss_curve.append(loss.cpu().detach().numpy())
            optimizer.step()

    plt.plot(loss_curve)
    plt.xlabel("Training steps")
    plt.ylabel("Loss")

Training iUNets for segmentation tasks
--------------------------------------

Unlike the above toy problem, most real-world tasks are inherently
non-invertible. This is in particular the case for segmentation problems,
where the number of input channels generally differs from the number of
classes (i.e. the number of output channels). In this case, one can simply use
e.g. convolutional layers as input and output layers to change the number of
channels to some desired number. Now the memory-efficient backpropagation is
automatically used in the invertible portions of the network, i.e. the iUNet.

.. code:: python

    import torch
    from torch import nn
    from iunets import iUNet

    INPUT_CHANNELS = 3
    INTERMEDIATE_CHANNELS = 64
    OUTPUT_CHANNELS = 10

    input_layer = nn.Conv3d(
        INPUT_CHANNELS,
        INTERMEDIATE_CHANNELS,
        kernel_size=3,
        padding=1
    )

    iunet = iUNet(
        in_channels=INTERMEDIATE_CHANNELS,
        dim=3,
        architecture=(2,3,4)
    )

    output_layer = nn.Conv3d(
        INTERMEDIATE_CHANNELS,
        OUTPUT_CHANNELS,
        kernel_size=3,
        padding=1
    )

    # Chain all sub-networks together
    model = nn.Sequential(input_layer, iunet, output_layer).to('cuda')

