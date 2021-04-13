===========================================
iUNets - Fully invertible U-Nets in Pytorch
===========================================

This library enables highly memory-efficient training of **fully-invertible
U-Nets (iUNets)** in 1D, 2D and 3D for use cases such as segmentation of medical
images. It is based on the paper
`iUNets: Fully invertible U-Nets with Learnable Up- and Downsampling
<https://arxiv.org/abs/2005.05220>`_ by Christian Etmann, Rihuan Ke &
Carola-Bibiane Schönlieb.

The library can be installed via the following command:

.. code-block:: text

    pip install iunets


Official documentation:
`https://iunets.readthedocs.io <https://iunets.readthedocs.io>`_.

By combining well-known reversible layers (such as additive coupling layers)
with novel **learnable invertible up- and downsampling operators** and suitable
channel splitting/concatenation, the iUNet is fully bijective. This allows
for *reconstructing* activations *instead of storing* them. As such, the
memory demand is (in theory) independent of the number of layers.

.. figure:: img/iunet_for_segmentation.png

    An iUNet used as a memory-efficient sub-network for segmenting a 3-channel
    input into 10 classes.

The following table exemplifies the memory savings that can be achieved
by applying our memory-efficient gradient calculation, in contrast to the
conventional backpropagation procedure. Details are found in the paper.


+-------+--------------+-------------+-------+
| Depth | Conventional |     Ours    | Ratio |
+=======+==============+=============+=======+
| 5     | 3.17 GB      | **0.85 GB** | 26.8% |
+-------+--------------+-------------+-------+
| 10    | 5.90 GB      | **1.09 GB** | 18.4% |
+-------+--------------+-------------+-------+
| 20    | 11.4 GB      | **1.57 GB** | 13.8% |
+-------+--------------+-------------+-------+
| 30    | 16.8 GB      | **2.06 GB** | 12.2% |
+-------+--------------+-------------+-------+

If you're using this code in a publication, please cite this as:

.. code-block:: text

    @inproceedings{etmann2020iunets,
      title={iUNets: learnable invertible up-and downsampling for large-scale inverse problems},
      author={Etmann, Christian and Ke, Rihuan and Sch{\"o}nlieb, Carola-Bibiane},
      booktitle={2020 IEEE 30th International Workshop on Machine Learning for Signal Processing (MLSP)},
      pages={1--6},
      year={2020},
      organization={IEEE}
    }

Features
--------

- Easily set up memory-efficient iUNets in 1D, 2D or 3D, that can be trained like any other model in Pytorch and can be used e.g. for high-dimensional segmentation.
- Highly customizable.
- Learnable, invertible and possibly anisotropic up- and downsampling.
- Orthogonal channel mixing.
- Orthogonality for channel mixing and learnable invertible up- and downsampling is enforced by efficient `Lie group` methods, i.e. `Cayley transforms` and `matrix exponentials` of skew-symmetric matrices.
- Quality-of-life features such as automatic, dynamic padding and unpadding (if required for invertibility), as well as model summaries.



Requirements
------------

iUNets are powered by the following two libraries:

- `Pytorch`_
- `MemCNN`_

.. _MemCNN: https://github.com/silvandeleemput/memcnn
.. _Pytorch: https://pytorch.org



Example usage
-------------

A simple 2D iUNet
^^^^^^^^^^^^^^^^^

A version of the iUNet depicted above can be created incredibly simply. Let's
say that we want 2 additive coupling layers per resolution, both in the
downsampling branch (left) and the upsampling branch (right).

.. code:: python

    from iunets import iUNet
    model = iUNet(
        channels=(64,128,256,384),
        architecture=(2,2,2,2),
        dim=2
    )
    model.print_layout()

Output:

.. code-block:: text

    64-64-(32/32)---------------------------------------------------------(32/32)-64-64
    ---------128-128-(64/64)----------------------------------(64/64)-128-128----------
    ---------------------256-256-(128/128)-------(128/128)-256-256---------------------
    ---------------------------------512-512--512-512----------------------------------

This model can now be integrated into the normal Pytorch workflow (and in
particular be used as a sub-network) just like any other `torch.nn.Module`,
and it automatically employs the memory-efficient backpropagation.

A fully-customized 3D iUNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the above example shows that a simple iUNet can be created quite simply,
our library also allows for a high degree of customization. Refer to the API
documentation for more information.

.. code:: python

    from iunets import iUNet
    from iunets.layers import create_standard_module
    model = iUNet(
        channels=(7,15,35,91),
        dim=3,
        architecture=[2,3,1,3],
        create_module_fn=create_standard_module,
        module_kwargs={'depth': 3},
        slice_mode='double',
        resampling_stride=[2,2,(1,2,2)],
        learnable_resampling=True,
        resampling_init='haar',
        resampling_method='cayley',
        disable_custom_gradient=False,
        revert_input_padding=True,
        padding_mode='reflect',
        verbose=1
        )
    model.print_layout()

Output:

.. code-block:: text

    Could not exactly create an iUNet with channels=(7, 15, 35, 91) and
    resampling_stride=[(2, 2, 2), (2, 2, 2), (1, 2, 2)]. Instead using closest
    achievable configuration: channels=[7, 16, 32, 92].
    Average relative error: 0.0508

    7-7-(5/2)-------------------------------------------------(5/2)-7-7
    ------16-16-16-(12/4)------------------------(12/4)-16-16-16-------
    ------------------32-(9/23)------------(9/23)-32-------------------
    ------------------------92-92-92--92-92-92-------------------------


