Migrating from 0.1
==================

The documented class is now ``IUNet``; ``iUNet`` remains an alias.
``disable_custom_gradient=False`` becomes ``gradient_mode="reconstruct"``
and ``disable_custom_gradient=True`` becomes ``gradient_mode="autograd"``.
Unknown constructor arguments now raise ``TypeError``. In particular, the old
example's ``slice_mode`` argument was ignored and should be removed.

``layout()`` returns a string for logging or display; ``print_layout()``
continues to print it.

The 0.2 package no longer depends on MemCNN. Parameters remain under the
``_fn`` checkpoint namespace, so ordinary 0.1 state dictionaries should load
with ``model.load_state_dict(checkpoint, strict=True)``. Older default blocks
registered their convolution layers twice as ``.F.seq.`` and ``.F.F.``.
Loading discards the duplicate ``.F.F.`` keys, or maps them to ``.F.seq.``
when needed. Custom module checkpoints still require the same module factory
and channel schedule. The internal ``iunets.cayley`` and ``iunets.expm``
modules were removed; orthogonal maps now use ``torch.linalg.solve`` and
``torch.linalg.matrix_exp`` inside ``iunets.layers``.

