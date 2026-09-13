Gradient modes
==============

``gradient_mode="reconstruct"`` is the default. The encoder and decoder are
each treated as a sequence of reversible operations. The engine saves only
their boundary tensors and trainable parameters for backward. It walks each
sequence in reverse, recovers the preceding state with ``inverse()``, and uses
``torch.func.vjp`` to compute local input and parameter gradients. The
activation storage for each sequence is independent of its number of blocks.
Reconstruction adds inverse and local forward work during backward, and
roundoff can accumulate in very deep networks.

``gradient_mode="autograd"`` uses ordinary PyTorch recording. It retains
intermediate activations, so memory use grows with network depth, but avoids
reconstruction work. Select it for a custom module with randomness, mutable
buffers, or a stateful inverse:

.. code-block:: python

   model = IUNet((4, 8), (2, 2), dim=2, gradient_mode="autograd")

Custom reversible modules
-------------------------

``create_module_fn(channels, dim=..., branch=..., level=...,
module_index=..., architecture=..., **module_kwargs)`` must return a
``torch.nn.Module`` with both ``forward(x)`` and ``inverse(y)``. Reconstruction
requires these methods to be deterministic, side effect free, and mutually
inverse for the same parameters. Dropout and normalization with running
statistics in training mode require ``gradient_mode="autograd"``. The built in
default coupling block uses instance normalization without running statistics.

Transforms and compilation
--------------------------

Reconstruction supports eager backpropagation, AMP, higher order gradients,
and ``torch.func`` transforms such as ``grad``, ``jvp``, ``vmap``, and
``hessian``. Full graph compiled training is supported with
``torch.compile(model, fullgraph=True)``. PyTorch currently rejects a custom
autograd function that defines ``jvp`` during compilation, so the library
selects a compile specific variant with the same backward reconstruction
rule. PyTorch 2.3 cannot trace reversible module calls inside a compiled
custom autograd function, so compiled training on 2.3 uses ordinary autograd
and retains intermediate activations. Eager training on 2.3 still uses
reconstruction. Nested composition of ``torch.compile`` with ``torch.func``
transforms is outside the 0.2 release scope.
