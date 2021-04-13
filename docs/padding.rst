=========================
Tutorial 5: Input Padding
=========================

For full invertibility of any invertible network, the total dimensionality of
the output has to be the same as the dimensionality of the input. This can be
problematic when invertible downsampling operations are involved. Take the
example of an image with odd-valued resolution -- this cannot be downsampled
exactly by a factor of 2 without remainder! Our library automatically chooses
the padding required to guarantee invertibility from padded input to output.

.. code:: python

    import torch
    from torch.nn.functional import pad
    from iunets import iUNet
    model = iUNet(
        channels=(32,64,128,256),
        architecture=(2,2,2,2),
        dim=2,
        resampling_stride=[2, (3,2), (5,3)],
        revert_input_padding=False,
        verbose=0,
    )
    x = torch.randn(2, 32, 128, 128)
    y = model(x)

    print("Input batch shape: {}".format(tuple(x.shape)))
    print("resampling_stride={} results in total downsampling factors per "
        "spatial dimension of {}".format(
            model.resampling_stride, model.downsampling_factors
        ))
    print("Trying to pad to resolution {}...".format(model.get_padding(x)[0]))
    print("Now the minimal feature resolution in the network will be {}".format(
        tuple(model.encode(x, use_padding=True)[-1].shape[-2:])))
    print("Padded output shape: {}".format(tuple(y.shape)))

    print("Setting revert_input_padding to False (standard behavior)...")
    model.revert_input_padding = True
    y = model(x)
    print("Now cropped output shape: {}".format(tuple(y.shape)))

Output:

.. code-block:: text

    [...]
    Input batch shape: (2, 32, 128, 128)
    resampling_stride=[(2, 2), (3, 2), (5, 3)] results in total downsampling factors per spatial dimension of (30, 12)
    Trying to pad to resolution [150, 132]...
    Now the minimal feature resolution in the network will be (5, 11)
    Padded output shape: (2, 32, 150, 132)
    Setting revert_input_padding to True (standard behavior)...
    Now cropped output shape: (2, 32, 128, 128)

This functionality uses the ``torch.nn.functional.pad`` API. For further control,
the user can set its ``padding_mode`` and ``padding_value`` parameters in the
iUNet constructor, e.g. ``padding_mode="reflect"``.
