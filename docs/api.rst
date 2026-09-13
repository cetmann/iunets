Python API
==========

.. autoclass:: iunets.IUNet
   :members: forward, inverse, encode, decode, encoder_inverse, decoder_inverse, layout, print_layout, get_padding, pad, revert_padding

.. autofunction:: iunets.layers.create_standard_module

.. autoclass:: iunets.layers.AdditiveCoupling

.. autoclass:: iunets.layers.InvertibleDownsampling1D
.. autoclass:: iunets.layers.InvertibleDownsampling2D
.. autoclass:: iunets.layers.InvertibleDownsampling3D
.. autoclass:: iunets.layers.InvertibleUpsampling1D
.. autoclass:: iunets.layers.InvertibleUpsampling2D
.. autoclass:: iunets.layers.InvertibleUpsampling3D
