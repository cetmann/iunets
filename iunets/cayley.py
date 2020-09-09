import torch
from torch.autograd import Function

def _cayley(A):
    I = torch.eye(A.shape[-1], device=A.device)
    LU  = torch.lu(I+A, pivot=True)
    return torch.lu_solve(I-A,*LU)

def _cayley_inverse(Q):
    I = torch.eye(Q.shape[-1], device=Q.device)
    rec_LU = torch.lu(I+Q, pivot=True)
    return torch.lu_solve(I-Q,*rec_LU)

def _cayley_frechet(A,H,Q=None):
    I = torch.eye(A.shape[-1], device=A.device)
    if Q is None:
        Q = _cayley(A)
    _LU = torch.lu(I+A, pivot=True)
    p = torch.lu_solve(Q, *_LU)
    _LU = torch.lu(I-A, pivot=True)
    q = torch.lu_solve(H, *_LU)
    return 2.* q @ p

class cayley(Function):
    """Computes the Cayley transform.

    """
    @staticmethod
    def forward(ctx, M):
        cayley_M = _cayley(M)
        ctx.save_for_backward(M, cayley_M)
        return cayley_M

    @staticmethod
    def backward(ctx, grad_out):
        M, cayley_M = ctx.saved_tensors
        dcayley_M = _cayley_frechet(M, grad_out, Q=cayley_M) 
        return dcayley_M
