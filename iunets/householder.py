import torch
from torch.autograd import Function
from .utils import eye_like
from warnings import warn

def householder_matrix(unit_vector):
    # If unit_vector has shape (batch,m) or (m,), turn it into matrix shape
    # (batch,m,1) respectively (m,1).
    if unit_vector.shape[-1] != 1:
        # Handle edge case if unit_vector.shape is (1,)
        if len(unit_vector.shape) == 1:
            return torch.ones_like(unit_vector)
        unit_vector = unit_vector.view(*tuple(unit_vector.shape), 1)
    transform = 2 * unit_vector @ torch.transpose(unit_vector, -1, -2)
    return eye_like(transform) - transform

def normalize_matrix_rows(matrix, eps=1e-6):
    norms = torch.sqrt(torch.sum(matrix**2, dim=-2, keepdim=True) + eps)
    return matrix / norms

def householder_transform(matrix, n_reflections=-1, eps=1e-6):
    """Implements a product of Householder transforms.

    """
    # If n_reflections==-1, use as many reflections as possible.
    if n_reflections == -1:
        n_reflections = matrix.shape[-1]
    if n_reflections > matrix.shape[-1]:
        warn("n_reflections is set higher than the number of rows.")
        n_reflections = matrix.shape[-1]
    matrix = normalize_matrix_rows(matrix, eps)
    if n_reflections == 0:
        output = torch.eye(
            matrix.shape[-2],
            dtype=matrix.dtype,
            device=matrix.device
        )
        if len(matrix.shape) == 3:
            output = output.view(1, matrix.shape[1], matrix.shape[1])
            output = output.expand(matrix.shape[0], -1, -1)

    for i in range(n_reflections):
        unit_vector = matrix[..., i:i+1]
        householder = householder_matrix(unit_vector)
        if i == 0:
            output = householder
        else:
            output = output @ householder
    return output