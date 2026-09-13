Quickstart
==========

Installation
------------

Install with ``pip install iunets``. For development, clone the repository and
run ``uv sync --frozen --extra docs``. The lockfile is for reproducible
development; the package metadata allows newer compatible PyTorch releases.

Train on CPU
------------

.. code-block:: python

   import torch
   from iunets import IUNet

   model = IUNet(channels=(4, 8), architecture=(2, 2), dim=2)
   images = torch.randn(2, 4, 16, 16)
   targets = torch.flip(images, dims=(-1,))
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   optimizer.zero_grad()
   prediction = model(images)
   loss = torch.nn.functional.mse_loss(prediction, targets)
   loss.backward()
   optimizer.step()

   print(model.layout())

Spatial dimensions divisible by the total downsampling stride allow an exact
round trip up to floating point error:

.. code-block:: python

   model.eval()
   with torch.no_grad():
       reconstructed = model.inverse(model(images))
   torch.testing.assert_close(reconstructed, images, atol=1e-4, rtol=1e-4)

Padding and channel layout
--------------------------

``channels`` gives the desired count at each level. Some schedules cannot be
represented exactly because each level must split channels and preserve the
total number of elements under downsampling. Inspect ``model.channels`` and
``model.layout()`` for the actual schedule.

The model pads spatial dimensions automatically when a stride does not divide
them. With ``revert_input_padding=True`` (the default), output is cropped to
the original shape. Cropping discards the padded border, so calling
``inverse(model(x))`` after such cropping need not reproduce ``x`` exactly.
Set ``revert_input_padding=False`` to retain the padded, invertible domain:

.. code-block:: python

   model = IUNet((4, 8), (1, 1), dim=2,
                 resampling_stride=(2, 1), revert_input_padding=False)
   x = torch.randn(1, 4, 5, 7)
   y = model(x)
   assert y.shape == (1, 4, 6, 7)
   torch.testing.assert_close(model.inverse(y), model.pad(x), atol=1e-4, rtol=1e-4)

The supported orthogonal methods are ``"cayley"``, ``"exp"``, and
``"householder"``. Choose with ``resampling_method`` and
``channel_mixing_method``. Householder maps use random initialization.
