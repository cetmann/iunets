import torch
from torch import nn, Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.autograd import Function

from typing import Callable, Union, Iterable, Tuple

import memcnn
from memcnn import InvertibleModuleWrapper, models
import numpy as np

from .utils import calculate_shapes_or_channels, get_num_channels
from .expm import expm
from .cayley import cayley


def __calculate_kernel_matrix_exp__(weight, *args, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return expm.apply(skew_symmetric_matrix)


def __calculate_kernel_matrix_cayley__(weight, *args, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return cayley.apply(skew_symmetric_matrix)


def __calculate_kernel_matrix_householder__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Householder transform " 
        "not implemented.")


def __calculate_kernel_matrix_givens__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Givens rotations not "
        "implemented.")


def __calculate_kernel_matrix_bjork__(weight, *args, **kwargs):
    raise NotImplementedError("Parametrization via Bjork peojections "
        "not implemented.")


def __initialize_weight__(kernel_matrix_shape : Tuple[int, ...],
                          stride : Tuple[int, ...],
                          method : str = 'exp',
                          init : str = 'haar',
                          dtype : str = 'float32',
                          *args,
                          **kwargs):
    """Function which computes specific orthogonal matrices.

    For some chosen method of parametrizing orthogonal matrices, this
    function outputs the required weights necessary to represent a
    chosen initialization as a Pytorch tensor of matrices.

    Args:
        kernel_matrix_shape : The output shape of the
            orthogonal matrices. Should be (num_matrices, height, width).
        stride : The stride for the invertible up- or
            downsampling for which this matrix is to be used. The length
            of ``stride`` should match the dimensionality of the data.
        method : The method for parametrising orthogonal matrices.
            Should be 'exp' or 'cayley'
        init : The matrix which should be represented. Should be
            'squeeze', 'pixel_shuffle', 'haar' or 'random'. 'haar' is only
            possible if ``stride`` is only 2.
        dtype : Numpy dtype which should be used for the matrix.
        *args: Variable length argument iterable.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor : Orthogonal matrices of shape ``kernel_matrix_shape``.
    """

    # Determine dimensionality of the data and the number of matrices.
    dim = len(stride)
    num_matrices = kernel_matrix_shape[0]
    
    # tbd: Givens, Householder, Bjork, give proper exception.
    assert(method in ['exp', 'cayley']) 
        
    if init == 'random':
        return torch.randn(kernel_matrix_shape)
    
    if init == 'haar' and set(stride) != {2}:
        print("Initialization 'haar' only available for stride 2.")
        print("Falling back to 'squeeze' transform...")
        init = 'squeeze'
    
    if init == 'haar' and set(stride) == {2}:
        if method == 'exp':
            # The following matrices each parametrize the Haar transform when
            # exponentiating the skew symmetric matrix weight-weight.T
            # Can be derived from the theory in Gallier, Jean, and Dianna Xu. 
            # "Computing exponentials of skew-symmetric matrices and logarithms
            # of orthogonal matrices." International Journal of Robotics and
            # Automation 18.1 (2003): 10-20.
            p = np.pi/4
            if dim == 1:
                weight = np.array([[[0, p],
                                    [0, 0]]],
                                  dtype=dtype)
            if dim == 2:
                weight = np.array([[[0, 0,  p,  p],
                                    [0, 0, -p, -p],
                                    [0, 0,  0,  0],
                                    [0, 0,  0,  0]]],
                                  dtype=dtype)
            if dim == 3:
                weight = np.array(
                    [[[0, p, p, 0, p, 0, 0, 0],
                      [0, 0, 0, p, 0, p, 0, 0],
                      [0, 0, 0, p, 0, 0, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, p, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, 0]]],
                    dtype=dtype)
            
            return torch.tensor(weight).repeat(num_matrices,1,1)

        elif method == 'cayley':
            # The following matrices parametrize a Haar kernel matrix
            # when applying the Cayley transform. These can be found by
            # applying an inverse Cayley transform to a Haar kernel matrix.
            if dim == 1:
                p = -np.sqrt(2)/(2-np.sqrt(2))
                weight = np.array([[[0, p],
                                    [0, 0]]],
                                  dtype=dtype)
            if dim == 2:
                p = .5
                weight = np.array([[[0, 0,  p,  p],
                                    [0, 0, -p, -p],
                                    [0, 0,  0,  0],
                                    [0, 0,  0,  0]]],
                                  dtype=dtype)
            if dim == 3:
                p=1/np.sqrt(2)
                weight = np.array(
                    [[[0, -p, -p,  0,  -p,   0,   0, 1-p],
                      [0,  0,  0, -p,   0,  -p, p-1,   0],
                      [0,  0,  0, -p,   0, p-1,  -p,   0],
                      [0,  0,  0,  0, 1-p,   0,   0,  -p],
                      [0,  0,  0,  0,   0,  -p,  -p,   0],
                      [0,  0,  0,  0,   0,   0,   0,  -p],
                      [0,  0,  0,  0,   0,   0,   0,  -p],
                      [0,  0,  0,  0,   0,   0,   0,   0]]],
                    dtype=dtype)
            return torch.tensor(weight).repeat(num_matrices,1,1)

    if init in ['squeeze', 'pixel_shuffle']:
        if method == 'exp' or method == 'cayley':
            return torch.zeros(*kernel_matrix_shape)

    # An initialization of the weight can also be explicitly provided as a
    # numpy or torch tensor. If only one matrix is provided, this matrix
    # is copied num_matrices times.
    if type(init) is np.ndarray:
        init = torch.tensor(np.cast(init, dtype))

    if torch.is_tensor(init):
        if len(init.shape) == 2:
            init = init.reshape(1, *init.shape)
        if init.shape[0] == 1:
            init = init.repeat(num_matrices,1,1)
        assert(init.shape == kernel_matrix_shape)
        return init

    else:
        raise NotImplementedError("Unknown initialization.")

class OrthogonalResamplingLayer(torch.nn.Module):
    """Base class for orthogonal up- and downsampling operators.

    Attributes:
        low_channel_number : Lower number of channels. These are the input
            channels in the case of downsampling ops, and the output
            channels in the case of upsampling ops.
        high_channel_number : Higher number of channels. These are the output
            channels in the case of downsampling ops, and the input
            channels in the case of upsampling ops.
        stride : The downsampling / upsampling factor for each dimension.
        channel_multiplier : The channel multiplier, i.e. the number
            by which the number of channels are multiplied (downsampling)
            or divided (upsampling).
        method : Which method to use for parametrizing orthogonal
            matrices which are used as convolutional kernels.
    """
    def __init__(self, 
                 low_channel_number : int,
                 stride : Union[int, Tuple[int, ...]],
                 method : str = 'exp',
                 init : Union[str, np.ndarray, torch.Tensor] = 'haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        
        super(OrthogonalResamplingLayer, self).__init__()
        self.low_channel_number = low_channel_number
        self.method = method
        self.stride = stride
        self.channel_multiplier = int(np.prod(stride))
        self.high_channel_number = self.channel_multiplier * low_channel_number
        
        assert(method in ['exp', 'cayley'])
        if method == 'exp':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ = __calculate_kernel_matrix_cayley__
        
        self._kernel_matrix_shape = ((self.low_channel_number,)
                                     + (self.channel_multiplier,) * 2)
        self._kernel_shape = ((self.channel_multiplier
                               * self.low_channel_number, 1)
                              + self.stride)

        self.weight = torch.nn.Parameter(
            __initialize_weight__(kernel_matrix_shape=self._kernel_matrix_shape, 
                                  stride=self.stride, 
                                  method=self.method, 
                                  init=init)
        )
        self.weight.requires_grad = learnable
        
        
    # Apply the chosen method to the weight in order to parametrize 
    # an orthogonal matrix, then reshape into a convolutional kernel.
    @property
    def kernel_matrix(self):
        return self.__calculate_kernel_matrix__(self.weight)
    
    @property
    def kernel(self):
        return self.kernel_matrix.reshape(*self._kernel_shape)   
    
    
    
class InvertibleDownsampling1D(OrthogonalResamplingLayer):
    def __init__(self, 
                 in_channels : int, 
                 stride : _size_1_t = 2, 
                 method : str = 'exp',
                 init : str = 'haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        stride = tuple(_single(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling1D, self).__init__(
            low_channel_number = self.in_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

    def forward(self, x):

        # Convolve with stride 2 in order to invertibly downsample.
        return F.conv1d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Apply transposed convolution in order to invert the downsampling.
        return F.conv_transpose1d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    
class InvertibleUpsampling1D(OrthogonalResamplingLayer):
    def __init__(self, 
                 in_channels : int, 
                 stride : _size_1_t = 2,
                 method : str = 'exp',
                 init : str = 'haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling1D, self).__init__(
            low_channel_number = self.out_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

    def forward(self, x):
        # Apply transposed convolution in order to invertibly upsample.
        return F.conv_transpose1d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Convolve with stride 2 in order to invert the upsampling.
        return F.conv1d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
       

class InvertibleDownsampling2D(OrthogonalResamplingLayer):
    def __init__(self,
                 in_channels : int,
                 stride : _size_2_t = 2,
                 method : str = 'exp',
                 init : str = 'haar',
                 learnable : bool = True,
                 *args,
                 **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling2D, self).__init__(
            low_channel_number = self.in_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

        
    
    def forward(self, x):
        # Convolve with stride 2 in order to invertibly downsample.
        return F.conv2d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Apply transposed convolution in order to invert the downsampling.
        return F.conv_transpose2d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    
class InvertibleUpsampling2D(OrthogonalResamplingLayer):
    def __init__(self, 
                 in_channels : int, 
                 stride : _size_2_t = 2, 
                 method : str = 'exp',
                 init : str = 'haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        stride = tuple(_pair(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling2D, self).__init__(
            low_channel_number = self.out_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

    def forward(self, x):
        # Apply transposed convolution in order to invertibly upsample.
        return F.conv_transpose2d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Convolve with stride 2 in order to invert the upsampling.
        return F.conv2d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    
    
    
class InvertibleDownsampling3D(OrthogonalResamplingLayer):
    def __init__(self, 
                 in_channels : int, 
                 stride : _size_3_t = 2, 
                 method : str = 'exp',
                 init : str = 'haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling3D, self).__init__(
            low_channel_number = self.in_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

    def forward(self, x):
        # Convolve with stride 2 in order to invertibly downsample.
        return F.conv3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Apply transposed convolution in order to invert the downsampling.
        return F.conv_transpose3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    
class InvertibleUpsampling3D(OrthogonalResamplingLayer):
    def __init__(self, 
                 in_channels : int, 
                 stride : _size_3_t = 2, 
                 method : str = 'exp',
                 init : str ='haar',
                 learnable : bool = True,
                 *args, 
                 **kwargs):
        stride = tuple(_triple(stride))
        channel_multiplier = int(np.prod(stride))
        self.in_channels = in_channels
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling3D, self).__init__(
            low_channel_number = self.out_channels,
            stride = stride, 
            method = method, 
            init = init, 
            learnable = learnable, 
            *args,
            **kwargs
        )

    def forward(self, x):
        # Apply transposed convolution in order to invertibly upsample.
        return F.conv_transpose3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)
    
    def inverse(self, x):
        # Convolve with stride 2 in order to invert the upsampling.
        return F.conv3d(
            x, self.kernel, stride=self.stride, groups=self.low_channel_number)









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
    
class StandardAdditiveCoupling(nn.Module):
    def __init__(self, F, channel_split_pos):
        """
        This computes the output :math:`y` on forward given input :math:`x`
        and arbitrary modules :math:`F` according to:
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

    

    
class StandardBlock(nn.Module):
    def __init__(self, dim, num_in_channels, num_out_channels, zero_init=True):
        super(StandardBlock, self).__init__()
        
        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        
        self.seq = nn.ModuleList()
        
        self.seq.append(conv_op(num_in_channels, num_out_channels, 3, padding=1, bias=False))
        torch.nn.init.kaiming_uniform_(self.seq[-1].weight, 
                                       a=0.01, 
                                       mode='fan_out', 
                                       nonlinearity='leaky_relu')
        
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
    

def create_standard_module(input_shape_or_channels, dim, LR, i, j, architecture, *args, **kwargs):
    num_channels = get_num_channels(input_shape_or_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels
    return StandardAdditiveCoupling(
        F=StandardBlock(dim, num_F_in_channels, num_F_out_channels), channel_split_pos=num_F_out_channels
        )
