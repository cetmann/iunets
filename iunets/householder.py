"""Utilities for orthogonal matrices built from Householder reflections."""

from warnings import warn

import torch

from .utils import eye_like


def householder_matrix(unit_vector):
    """Return the reflection matrix for one or more unit vectors.

    ``unit_vector`` may have shape ``(n,)``, ``(..., n)``, or ``(..., n, 1)``.
    The final singleton dimension is added when needed so batched matrix
    multiplication remains broadcastable. The input is expected to be
    normalized by the caller.
    """

    if unit_vector.shape[-1] != 1:
        unit_vector = unit_vector.unsqueeze(-1)
    transform = 2 * unit_vector @ torch.transpose(unit_vector, -1, -2)
    return eye_like(transform) - transform


def normalize_matrix_rows(matrix, eps=1e-6):
    """Normalize matrix columns, guarding against zero columns.

    The historical function name is kept for compatibility; normalization is
    along the row axis because each column is a reflection vector.
    """

    norms = torch.linalg.vector_norm(matrix, dim=-2, keepdim=True).clamp_min(eps)
    return matrix / norms


def householder_transform(matrix, n_reflections=-1, eps=1e-6):
    """Construct an orthogonal matrix as a product of Householder reflections.

    Args:
        matrix: Tensor whose columns provide the reflection vectors. Its shape
            is ``(n, n)`` or ``(batch, n, n)``.
        n_reflections: Number of columns/reflections to apply. ``-1`` uses all
            available columns.
        eps: Minimum norm used while normalizing reflection vectors.
    """

    if n_reflections == -1:
        n_reflections = matrix.shape[-1]
    if n_reflections > matrix.shape[-1]:
        warn("n_reflections is set higher than the number of columns.")
        n_reflections = matrix.shape[-1]
    matrix = normalize_matrix_rows(matrix, eps)
    if n_reflections == 0:
        output = torch.eye(matrix.shape[-2], dtype=matrix.dtype, device=matrix.device)
        if len(matrix.shape) == 3:
            output = output.view(1, matrix.shape[1], matrix.shape[1])
            output = output.expand(matrix.shape[0], -1, -1)

    for i in range(n_reflections):
        unit_vector = matrix[..., i : i + 1]
        householder = householder_matrix(unit_vector)
        if i == 0:
            output = householder
        else:
            output = output @ householder
    return output
