import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import memcnn
from memcnn import InvertibleModuleWrapper, models
import torch
import warnings

# Load the C++ extension which wraps the cuDNN implementation of the adjoint of convolutions
# 
print("Compiling cuDNN convolution adjoint op...")
import os 
current_folder = dir_path = os.path.dirname(os.path.realpath(__file__))
from torch.utils.cpp_extension import load
cudnn_convolution = load(name="cudnn_convolution", sources=[current_folder+"/conv_ops.cpp"], verbose=True)
conv_weight = cudnn_convolution.convolution_backward_weight
print("Done.")

from .utils import calculate_shapes_or_channels, get_num_channels


def haar_kernel(dim):
    '''
    Outputs the filter bank for a haar-based invertible downsampling layer.
    The same kernel can be used for the invertible upsampling, using the 
    transposed convolution. Works for any data format (1D, 2D, 3D, 543D,...)
    '''
    
    # Implementation details: The n-dimensional haar filter bank is realized as
    # outer products of the haar scaling function Phi and the haar wavelet Psi.
    # This is realized by constructing bit sequences which determine the 
    # factors of the outer products.

    channel_factor = 2**dim
    
    multiplier = 2**(-(dim/2.))
    
    def exchange(t,i,j):
        # Returns a tuple with exchanged indices i and j
        out = list(t)
        out[i] = t[j]
        out[j] = t[i]
        return tuple(out)
    
    
    kernel_atom_shape = ()
    wavelet_shape = ()
    for i in range(dim):
        kernel_atom_shape+= (2,)
        wavelet_shape+= ((i==0)+1,)
     
    # Approximation function and wavelet
    Phi = torch.tensor([1,1])
    Psi = torch.tensor([1,-1])
    
    d = {0:Phi, 1:Psi}
    
    def wav(atom, j):
        result = d[atom].reshape(exchange(wavelet_shape, 0, j))
        return result
        
    kernel_atoms = []
    for i in range(channel_factor):
        kernel_atom = torch.zeros(kernel_atom_shape)
        indices_bitstring = ('{0:0'+str(dim)+'b}').format(i)
        indices = ()
        for j in range(len(indices_bitstring)):
            indices+= (int(indices_bitstring[j]),)
        kernel_atom = 1.
        for j in range(dim):
            kernel_atom = kernel_atom * wav(indices[j],j)
        kernel_atoms.append(multiplier*kernel_atom)
       
    kernel = torch.stack(kernel_atoms, dim=0).view(channel_factor,1,*([2]*dim))
    return kernel



class SplitChannels(torch.nn.Module):
    def __init__(self, split_location):
        super(SplitChannels, self).__init__()
        self.split_location = split_location
    
    def forward(self, x):
        a, b = (x[:,:self.split_location], 
                x[:,self.split_location:])
        a, b = a.clone(), b.clone()
        del x
        return a, b
    
    def inverse(self, x, y):
        return torch.cat([x, y], dim=1)
    

    
class ConcatenateChannels(torch.nn.Module):
    def __init__(self, split_location):
        super(ConcatenateChannels, self).__init__()
        self.split_location = split_location
    
    def forward(self, x, y):
        return torch.cat([x, y], dim=1)
    
    def inverse(self, x):
        a, b = (x[:,:self.split_location], 
                x[:,self.split_location:])
        a, b = a.clone(), b.clone()
        del x
        return a, b
    


def make_orthogonal_kernel(C, k):
    kernel_matrix = expm(C - torch.transpose(C,1,2))

    if C.shape[-1] == 2:
        kernel = kernel_matrix.view(k,1,2)
    elif C.shape[-1] == 4:
        kernel = kernel_matrix.view(k,1,2,2)
    elif C.shape[-1] == 8:
        kernel = kernel_matrix.view(k,1,2,2,2)
    else:
        raise NotImplementedError("Only 1D, 2D or 3D convolutional kernels are accepted.")
    return kernel


def expm(M, min_terms=30, *args, **kwargs):
    # Calculates the matrix exponential, works 'batch-wise' with broadcasting
    size = M.shape[-1]
    result = torch.eye(size, device=M.device).reshape(1,size,size)
    if len(M.shape) > 2:
        result = result.repeat(M.shape[0], 1, 1)
    matrix_result = result
    k = 1
    k_max = min_terms
    while k < k_max:
        matrix_result = torch.matmul(matrix_result, M) / k
        result = result + matrix_result
        k += 1
    return result

def expm_frechet(A, E, adjoint=False):
    # The adjoint of the Frechet derivative of the matrix exponential
    # of matrix A the Frechet derivative of the transpose of A
    if adjoint:
        A = A.transpose(-1,-2)    
    
    # Calculate the Frechet derivative via a power series
    size = A.shape[-1]
    result = E
    B = E
    P = E 
    for k in range(2,31):
        B = (1/k) * torch.matmul(A,B)
        P = (1/k) * torch.matmul(P,A) + B
        result = result + P
    return result

class invertible_downsampling(Function):
    @staticmethod
    def forward(ctx, C, x):
        d = len(x.shape)-2
        assert(d in [1,2,3])
        
        n_channels = x.shape[1]
        kernel = make_orthogonal_kernel(C, (2**d)*n_channels)
        
        conv_op = [F.conv1d, F.conv2d, F.conv3d][d-1]
        
        y = conv_op(x, 
                 kernel, 
                 bias=None,
                 stride=2,
                 groups=n_channels)
        ctx.save_for_backward(C, kernel, x) 
        return y
    
    
    @staticmethod
    def backward(ctx, grad_y):
        C, kernel, x = ctx.saved_tensors
        d = len(grad_y.shape)-2
        assert(d in [1,2,3])
        
        # If CUDA is not available, use Pytorch's device-agnostic
        # implementation of the weight-adjoint of the convolution.
        # If CUDA is available, use the cuDNN implementation, which
        # has a favorable memory footprint. 
        if not grad_y.is_cuda:
            warnings.warn("Not using CUDA for derivative of invertible downsampling.")
            conv_adjoint_weight_op = [F.grad.conv1d_weight, 
                                      F.grad.conv2d_weight, 
                                      F.grad.conv3d_weight][d-1]
            grad_kernel = conv_adjoint_weight_op(
                x, kernel.shape, grad_y, groups=x.shape[1])
        else:        
            stride = ([2]*(d))
            padding = ([0]*(d))
            dilation = ([1]*(d))
            grad_kernel = conv_weight(x, 
                                      kernel.shape, 
                                      grad_y, 
                                      stride, 
                                      padding, 
                                      dilation, 
                                      x.shape[1], 
                                      False, 
                                      True)
        
        grad_kernel = grad_kernel.reshape(-1, 2**d, 2**d)
        
        # Adjoint of frechet derivative of the exponential of C-C.T 
        # applied to kernel_grad
        grad_exp = expm_frechet(
            C-C.transpose(-1,-2),
            grad_kernel, 
            adjoint=True)
        grad_matrix = (grad_exp - grad_exp.transpose(-1,-2))

        
        conv_adjoint_input_op = [F.grad.conv1d_input,
                                 F.grad.conv2d_input,
                                 F.grad.conv3d_input][d-1]
        grad_x = conv_adjoint_input_op(
            x.shape, 
            kernel, 
            grad_y, 
            stride=2, 
            padding=0, 
            groups=x.shape[1])
        
        return grad_matrix, grad_x
    
    
class invertible_upsampling(Function):
    @staticmethod
    def forward(ctx, C, x):
        n_channels = x.shape[1]
        d = len(x.shape)-2
        assert(d in [1,2,3])
        
        kernel = make_orthogonal_kernel(C, n_channels)
        
        conv_transpose_op = [F.conv_transpose1d,
                             F.conv_transpose2d,
                             F.conv_transpose3d][d-1]
        y = conv_transpose_op(x, 
                 kernel, 
                 bias=None,
                 stride=2,
                 groups=n_channels//(2**d))
        ctx.save_for_backward(C, kernel, x) 
        return y

    @staticmethod
    def backward(ctx, grad_y):
        C, kernel, x = ctx.saved_tensors
        d = len(grad_y.shape)-2
        assert(d in [1,2,3])
        
        # If CUDA is not available, use Pytorch's device-agnostic
        # implementation of the weight-adjoint of the convolution.
        # If CUDA is available, use the cuDNN implementation, which
        # has a favorable memory footprint. 
        if not grad_y.is_cuda:
            warnings.warn("Not using CUDA for derivative of invertible upsampling.")
            conv_adjoint_weight_op = [F.grad.conv1d_weight, 
                                      F.grad.conv2d_weight, 
                                      F.grad.conv3d_weight][d-1]
            grad_kernel = conv_adjoint_weight_op(
                grad_y, kernel.shape, x, groups=grad_y.shape[1],stride=2,padding=0)
        else:
            stride = [2]*d
            padding = [0]*d
            dilation = [1]*d
            grad_kernel = conv_weight(grad_y, 
                                      kernel.shape, 
                                      x, 
                                      stride, 
                                      padding, 
                                      dilation, 
                                      grad_y.shape[1], 
                                      False, 
                                      True)
    
        grad_kernel = grad_kernel.reshape(-1, 2**d, 2**d)
        
        # adjoint of frechet derivative of the exponential of C-C.T 
        # applied to kernel_grad
        grad_exp = expm_frechet(
            C-C.transpose(-1,-2),
            grad_kernel, 
            adjoint=True)
        grad_matrix = (grad_exp - grad_exp.transpose(-1,-2))

        conv_op = [F.conv1d, F.conv2d, F.conv3d][d-1]
        grad_x = conv_op(
            grad_y,
            kernel, 
            stride=2, 
            padding=0, 
            groups=grad_y.shape[1])
        return grad_matrix, grad_x
    
    

class InvertibleDownsampling1D(torch.nn.Module):
    def __init__(self, input_channels=1, learnable=True):
        super(InvertibleDownsampling1D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels,2,2))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer(
                'kernel',
                haar_kernel(1).repeat(input_channels, 1, 1))
    
    def forward(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv1d(x, self.kernel, stride=2, groups=self.input_channels)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose1d(x, self.kernel, stride=2, groups=self.input_channels)
    

class InvertibleUpsampling1D(torch.nn.Module):
    def __init__(self, input_channels=2, learnable=True):
        super(InvertibleUpsampling2D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels//2, 2, 2))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer('kernel',
                                 haar_kernel(1).repeat(input_channels//2, 1, 1))
    
    def forward(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose1d(x, self.kernel, stride=2, groups=self.input_channels//2)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv1d(x, self.kernel, stride=2, groups=self.input_channels//2)
        
        
        
        
class InvertibleDownsampling2D(torch.nn.Module):
    def __init__(self, input_channels=1, learnable=True):
        super(InvertibleDownsampling2D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels,4,4))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer(
                'kernel',
                haar_kernel(2).repeat(input_channels, 1, 1, 1))
            
    
    def forward(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv2d(x, self.kernel, stride=2, groups=self.input_channels)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose2d(x, self.kernel, stride=2, groups=self.input_channels)
    

class InvertibleUpsampling2D(torch.nn.Module):
    def __init__(self, input_channels=4, learnable=True):
        super(InvertibleUpsampling2D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels//4, 4, 4))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer('kernel',
                                 haar_kernel(2).repeat(input_channels//4, 1, 1, 1))
    
    def forward(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose2d(x, self.kernel, stride=2, groups=self.input_channels//4)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv2d(x, self.kernel, stride=2, groups=self.input_channels//4)
    
    
    
    
    
class InvertibleDownsampling3D(torch.nn.Module):
    def __init__(self, input_channels=1, learnable=True):
        super(InvertibleDownsampling3D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels,8,8))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer(
                'kernel',
                haar_kernel(3).repeat(input_channels, 1, 1, 1, 1))
            
    
    def forward(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv3d(x, self.kernel, stride=2, groups=self.input_channels)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose3d(x, self.kernel, stride=2, groups=self.input_channels)        



class InvertibleUpsampling3D(torch.nn.Module):
    def __init__(self, input_channels=8, learnable=True):
        super(InvertibleUpsampling3D, self).__init__()
        self.learnable = learnable
        self.input_channels = input_channels
        if self.learnable:
            self.kernel_matrix = torch.nn.Parameter(torch.randn(input_channels//8, 8, 8))
            self.kernel_matrix.requires_grad = True
        else:
            self.register_buffer('kernel',
                                 haar_kernel(3).repeat(input_channels//8, 1, 1, 1, 1))
    
    def forward(self, x):
        if self.learnable:
            return invertible_upsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv_transpose3d(x, self.kernel, stride=2, groups=self.input_channels//8)
    
    def inverse(self, x):
        if self.learnable:
            return invertible_downsampling.apply(self.kernel_matrix, x)
        else:
            return F.conv3d(x, self.kernel, stride=2, groups=self.input_channels//8)
    
    
    
class StandardAdditiveCoupling(nn.Module):
    def __init__(self, F, channel_split_pos):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`F` according to:
        :math:`(x1, x2) = x`
        :math:`y1 = x2`
        :math:`y2 = x1 + F(y2)`
        :math:`y = (y1, y2)`
        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
        """
        super(StandardAdditiveCoupling, self).__init__()
        self.F = F
        self.channel_split_pos = channel_split_pos
        
    def forward(self, x):
        x1, x2 = x[:,:self.channel_split_pos], x[:,self.channel_split_pos:]
        x1, x2 = x1.contiguous(), x2.contiguous()
        y1 = x2
        y2 = x1 + self.F.forward(x2)
        out = torch.cat([y1, y2], dim=1)
        return out

    def inverse(self, y):
        #y1, y2 = torch.chunk(y, 2, dim=1)
        inverse_channel_split_pos = y.shape[1] - self.channel_split_pos
        y1, y2 = y[:,:inverse_channel_split_pos], y[:,inverse_channel_split_pos:]
        y1, y2 = y1.contiguous(), y2.contiguous()
        x2 = y1
        x1 = y2 - self.F.forward(y1)
        x = torch.cat([x1, x2], dim=1)
        return x

    
class Stable1x1Block(nn.Module):
    def __init__(self, dim, num_channels):
        super(Stable1x1Block, self).__init__()
        
        self.dim = dim
        self.num_channels = num_channels
        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv = conv_op(num_channels, num_channels, 1, padding=0, bias=False)
        
        self.scaling_fn = nn.Tanh()
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.multiplier = torch.nn.Parameter(torch.zeros(num_channels))
        
        # At initialization, perform an SVD and normalize the convolutional layer by
        # its spectral norm. Save the leading singular vector
        U, D, V = torch.svd(self.conv.weight.view(num_channels, num_channels))
        self.conv.register_buffer("singular_vec", U[:,0])
        with torch.no_grad():
            self.conv.weight /= D[0]
        del U, D, V
        
        # One power iteration is performed after each gradient step. The convolutional
        # kernel is normalized according to the spectral norm.
        def spectral_normalization_iteration(self, input): 
            kernel_matrix = self.weight.view(num_channels, num_channels)
            # self.singular_vec is unit-norm vector
            with torch.no_grad():
                singular_vec = kernel_matrix @ self.singular_vec
                singular_val = torch.norm(singular_vec)
                self.weight.data = self.weight.data / singular_val  
                self.singular_vec /= torch.norm(self.singular_vec)
            
        self.conv.register_forward_pre_hook(spectral_normalization_iteration)
        
    def forward(self, x):
        x = self.lrelu(x)
        x = self.conv(x)
        x = self.scaling_fn(self.multiplier).view(-1, *([1]*self.dim)) * x 
        x = x + self.bias.view(-1, *([1]*self.dim))
        return x
    
class StandardBlock(nn.Module):
    def __init__(self, dim, num_in_channels, num_out_channels, zero_init=True):
        super(StandardBlock, self).__init__()
        
        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        
        self.seq = nn.ModuleList()
        
        self.seq.append(conv_op(num_in_channels, num_out_channels, 3, padding=1, bias=False))
        self.seq.append(nn.LeakyReLU(inplace=True))
        # With groups=1, group normalization becomes layer normalization
        self.seq.append(nn.GroupNorm(1, num_out_channels, eps=1e-3)) 
        
        # Initialize the block as the zero transform, such that the coupling becomes
        # the coupling becomes an identity transform (up to permutation of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)
            
        
        self.F = nn.Sequential(*self.seq)
        
    def forward(self, x):
        return self.F(x)
    
    
class DoubleBlock(nn.Module):
    def __init__(self, dim, num_in_channels, num_out_channels, zero_init=True):
        super(DoubleBlock, self).__init__()
        
        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        
        self.seq = nn.ModuleList()
        
        self.seq.append(conv_op(num_in_channels, num_out_channels, 3, padding=1, bias=False))
        self.seq.append(nn.LeakyReLU(inplace=True))
        self.seq.append(nn.GroupNorm(1, num_out_channels, eps=1e-3)) 
        
        self.seq.append(conv_op(num_out_channels, num_out_channels, 3, padding=1, bias=False))
        self.seq.append(nn.LeakyReLU(inplace=True))
        self.seq.append(nn.GroupNorm(1, num_out_channels, eps=1e-3)) 
        # Initialize the block as the zero transform, such that the coupling becomes
        # the coupling becomes an identity transform (up to permutation of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)
        
        self.F = nn.Sequential(*self.seq)
        
    def forward(self, x):
        return self.F(x)    


def create_standard_module(input_shape_or_channels, dim, LR, i, j, architecture, *args, **kwargs):
    num_channels = get_num_channels(input_shape_or_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels
    return StandardAdditiveCoupling(
        F=StandardBlock(dim, num_F_in_channels, num_F_out_channels), channel_split_pos=num_F_out_channels
        )


def create_double_module(input_shape_or_channels, dim, LR, i, j, architecture, *args, **kwargs):
    num_channels = get_num_channels(input_shape_or_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels
    return StandardAdditiveCoupling(
        F=DoubleBlock(dim, num_F_in_channels, num_F_out_channels), channel_split_pos=num_F_out_channels
        )

def create_stable_1x1_module(input_shape_or_channels, dim, LR, i, j, architecture, *args, **kwargs):
    num_channels = get_num_channels(input_shape_or_channels)
    num_Fm_channels = num_channels // 2
    num_Gm_channels = num_channels - num_Fm_channels
    return StandardAdditiveCoupling(
        F=Stable1x1Block(dim, num_Fm_channels), dim=dim, num_channels=num_channels
        )
