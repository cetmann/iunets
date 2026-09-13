import math
from warnings import warn
from typing import Callable, Union, Iterable, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import get_num_channels
from .householder import householder_transform


def _stride_tuple(value, dim):
    if type(value) is int:
        value = (value,) * dim
    elif isinstance(value, (tuple, list)):
        value = tuple(value)
    else:
        raise ValueError(f"stride must be a positive integer or {dim} integers.")
    if len(value) != dim or any(type(part) is not int or part <= 0 for part in value):
        raise ValueError(f"stride must contain {dim} positive integers.")
    return value


def __calculate_kernel_matrix_exp__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    return torch.linalg.matrix_exp(skew_symmetric_matrix)


def __calculate_kernel_matrix_cayley__(weight, **kwargs):
    skew_symmetric_matrix = weight - torch.transpose(weight, -1, -2)
    identity = torch.eye(weight.shape[-1], dtype=weight.dtype, device=weight.device)
    return torch.linalg.solve(identity + skew_symmetric_matrix,
                              identity - skew_symmetric_matrix)


def __calculate_kernel_matrix_householder__(weight, **kwargs):
    n_reflections = kwargs.get('n_reflections', -1)
    eps = kwargs.get('eps', 1e-6)
    return householder_transform(weight, n_reflections, eps)


def __calculate_kernel_matrix_givens__(weight, **kwargs):
    raise NotImplementedError("Parametrization via Givens rotations not "
        "implemented.")


def __calculate_kernel_matrix_bjork__(weight, **kwargs):
    raise NotImplementedError("Parametrization via Bjork peojections "
        "not implemented.")



class OrthogonalResamplingLayer(torch.nn.Module):
    """Base class for orthogonal up- and downsampling operators.

    :param low_channel_number:
        Lower number of channels. These are the input
        channels in the case of downsampling ops, and the output
        channels in the case of upsampling ops.
    :param stride:
        The downsampling / upsampling factor for each dimension.
    :param channel_multiplier:
        The channel multiplier, i.e. the number
        by which the number of channels are multiplied (downsampling)
        or divided (upsampling).
    :param method:
        Which method to use for parametrizing orthogonal
        matrices which are used as convolutional kernels.
    """

    def __init__(self,
                 low_channel_number: int,
                 stride: Union[int, Tuple[int, ...]],
                 method: str = 'cayley',
                 init: Union[str, torch.Tensor] = 'haar',
                 learnable: bool = True,
                 init_kwargs: dict = None,
                 **kwargs):

        super(OrthogonalResamplingLayer, self).__init__()
        self.low_channel_number = low_channel_number
        self.method = method
        self.stride = stride
        self.channel_multiplier = math.prod(stride)
        self.high_channel_number = self.channel_multiplier * low_channel_number

        if init_kwargs is None:
            init_kwargs = {}
        self.init_kwargs = init_kwargs
        self.kwargs = kwargs

        if type(low_channel_number) is not int or low_channel_number <= 0:
            raise ValueError("low_channel_number must be a positive integer.")
        if method not in ('exp', 'cayley', 'householder'):
            raise ValueError("method must be 'exp', 'cayley', or 'householder'.")
        if method == 'exp':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_cayley__
        elif method == 'householder':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_householder__

        self._kernel_matrix_shape = ((self.low_channel_number,)
                                     + (self.channel_multiplier,) * 2)
        self._kernel_shape = ((self.high_channel_number, 1)
                              + self.stride)

        self.weight = torch.nn.Parameter(
            __initialize_weight__(kernel_matrix_shape=self._kernel_matrix_shape,
                                  stride=self.stride,
                                  method=self.method,
                                  init=init,
                                  **self.init_kwargs
                                  )
        )
        self.weight.requires_grad = learnable

    # Apply the chosen method to the weight in order to parametrize
    # an orthogonal matrix, then reshape into a convolutional kernel.
    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight, **self.kwargs)

    @property
    def kernel(self):
        """The kernel associated with the invertible up-/downsampling.
        """
        return self.kernel_matrix.reshape(*self._kernel_shape)


class InvertibleDownsampling1D(OrthogonalResamplingLayer):
    def __init__(self,
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 1)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling1D, self).__init__(
            low_channel_number=self.in_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 1)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        if in_channels % channel_multiplier:
            raise ValueError("in_channels must be divisible by the stride product.")
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling1D, self).__init__(
            low_channel_number=self.out_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 2)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling2D, self).__init__(
            low_channel_number=self.in_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 2)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        if in_channels % channel_multiplier:
            raise ValueError("in_channels must be divisible by the stride product.")
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling2D, self).__init__(
            low_channel_number=self.out_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 3)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        self.out_channels = in_channels * channel_multiplier
        super(InvertibleDownsampling3D, self).__init__(
            low_channel_number=self.in_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
                 in_channels: int,
                 stride: Union[int, Tuple[int, ...]] = 2,
                 method: str = 'cayley',
                 init: str = 'haar',
                 learnable: bool = True,
                 *args,
                 **kwargs):
        stride = _stride_tuple(stride, 3)
        channel_multiplier = math.prod(stride)
        self.in_channels = in_channels
        if in_channels % channel_multiplier:
            raise ValueError("in_channels must be divisible by the stride product.")
        self.out_channels = in_channels // channel_multiplier
        super(InvertibleUpsampling3D, self).__init__(
            low_channel_number=self.out_channels,
            stride=stride,
            method=method,
            init=init,
            learnable=learnable,
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
        a, b = (x[:, :self.split_location],
                x[:, self.split_location:])
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
        a, b = (x[:, :self.split_location],
                x[:, self.split_location:])
        a, b = a.clone(), b.clone()
        del x
        return a, b

class AdditiveCoupling(nn.Module):
    """Additive coupling layer, a basic invertible layer.

    By splitting the input activation :math:`x` and output activation :math:`y`
    into two groups of channels (i.e. :math:`(x_1, x_2) \\cong x` and
    :math:`(y_1, y_2) \\cong y`), `additive coupling layers` define an invertible
    mapping :math:`x \\mapsto y` via

    .. math::

       y_1 &= x_2

       y_2 &= x_1 + F(x_2),

    where the `coupling function` :math:`F` is an (almost) arbitrary mapping.
    :math:`F` just has to map from the space of :math:`x_2` to the space of
    :math:`x_1`. In practice, this can for instance be a sequence of
    convolutional layers with batch normalization.

    The inverse of the above mapping is computed algebraically via

    .. math::

       x_1 &= y_2 - F(y_1)

       x_2 &= y_1.

    *Warning*: Note that this is different from the definition of additive
    coupling layers in ``MemCNN``. Those are equivalent to two consecutive
    instances of the above-defined additive coupling layers. Hence, the
    variant implemented here is twice as memory-efficient as the variant from
    ``MemCNN``.

    :param F:
        The coupling function of the additive coupling layer, typically a
        sequence of neural network layers.
    :param channel_split_pos:
        The index of the channel at which the input and output activations are
        split.

    """
    def __init__(self,
                 F: nn.Module,
                 channel_split_pos: int):

        super(AdditiveCoupling, self).__init__()
        self.F = F
        self.channel_split_pos = channel_split_pos

    def forward(self, x):
        x1, x2 = x[:, :self.channel_split_pos], x[:, self.channel_split_pos:]
        x1, x2 = x1.contiguous(), x2.contiguous()
        y1 = x2
        y2 = x1 + self.F.forward(x2)
        out = torch.cat([y1, y2], dim=1)
        return out

    def inverse(self, y):
        # y1, y2 = torch.chunk(y, 2, dim=1)
        inverse_channel_split_pos = y.shape[1] - self.channel_split_pos
        y1, y2 = (y[:, :inverse_channel_split_pos],
                  y[:, inverse_channel_split_pos:])
        y1, y2 = y1.contiguous(), y2.contiguous()
        x2 = y1
        x1 = y2 - self.F.forward(y1)
        x = torch.cat([x1, x2], dim=1)
        return x


class StandardBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_in_channels,
                 num_out_channels,
                 depth=2,
                 zero_init=False,
                 normalization="instance",
                 **kwargs):
        super(StandardBlock, self).__init__()

        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        if type(depth) is not int or depth <= 0:
            raise ValueError("depth must be a positive integer.")
        if normalization not in ('instance', 'group', 'batch', None):
            raise ValueError("normalization must be 'instance', 'group', 'batch', or None.")

        for i in range(depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)

            if i == 0:
                current_in_channels = num_in_channels
            if i == depth-1:
                current_out_channels = num_out_channels

            self.seq.append(
                conv_op(
                    current_in_channels,
                    current_out_channels,
                    3,
                    padding=1,
                    bias=False))
            torch.nn.init.kaiming_uniform_(self.seq[-1].weight,
                                           a=0.01,
                                           mode='fan_out',
                                           nonlinearity='leaky_relu')

            if normalization == "instance":
                norm_op = [nn.InstanceNorm1d,
                           nn.InstanceNorm2d,
                           nn.InstanceNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, affine=True))

            elif normalization == "group":
                self.seq.append(
                    nn.GroupNorm(
                        max(group for group in range(1, min(8, current_out_channels) + 1)
                            if current_out_channels % group == 0),
                        current_out_channels,
                        affine=True)
                )

            elif normalization == "batch":
                norm_op = [nn.BatchNorm1d,
                           nn.BatchNorm2d,
                           nn.BatchNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, eps=1e-3))

            self.seq.append(nn.LeakyReLU(inplace=True))


        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            final_conv = next(layer for layer in reversed(self.seq)
                              if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))
            torch.nn.init.zeros_(final_conv.weight)

        self.seq = nn.Sequential(*self.seq)

    def forward(self, x):
        return self.seq(x)




def create_standard_module(in_channels, **kwargs):
    dim = kwargs.pop('dim', 2)
    depth = kwargs.pop('depth', 2)
    num_channels = get_num_channels(in_channels)
    num_F_in_channels = num_channels // 2
    num_F_out_channels = num_channels - num_F_in_channels

    module_index = kwargs.pop('module_index', 0)
    # For odd number of channels, this switches the roles of input and output
    # channels at every other layer, e.g. 1->2, then 2->1.
    if module_index % 2 == 0:
        (num_F_in_channels, num_F_out_channels) = (
            num_F_out_channels, num_F_in_channels
        )
    return AdditiveCoupling(
        F=StandardBlock(
            dim,
            num_F_in_channels,
            num_F_out_channels,
            depth=depth,
            **kwargs),
        channel_split_pos=num_F_out_channels
    )


def __initialize_weight__(kernel_matrix_shape : Tuple[int, ...],
                          stride : Tuple[int, ...],
                          method : str = 'cayley',
                          init : str = 'haar',
                          dtype : torch.dtype = torch.float32,
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
        dtype : PyTorch dtype which should be used for the matrix.
        *args: Variable length argument iterable.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor : Orthogonal matrices of shape ``kernel_matrix_shape``.
    """

    # Determine dimensionality of the data and the number of matrices.
    dim = len(stride)
    num_matrices = kernel_matrix_shape[0]
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if method not in ('exp', 'cayley', 'householder'):
        raise ValueError("method must be 'exp', 'cayley', or 'householder'.")
    if method == 'householder':
        if isinstance(init, str) and init != 'random':
            warn("Householder resampling supports random initialization; "
                 "using random weights.")
            init = 'random'

    if isinstance(init, str) and init == 'random':
        return torch.randn(kernel_matrix_shape, dtype=dtype)
    
    if isinstance(init, str) and init == 'haar' and set(stride) != {2}:
        warn("Haar initialization requires stride 2 in every dimension; "
             "using squeeze initialization.")
        init = 'squeeze'
    
    if isinstance(init, str) and init == 'haar' and set(stride) == {2}:
        if method == 'exp':
            # The following matrices each parametrize the Haar transform when
            # exponentiating the skew symmetric matrix weight-weight.T
            # Can be derived from the theory in Gallier, Jean, and Dianna Xu. 
            # "Computing exponentials of skew-symmetric matrices and logarithms
            # of orthogonal matrices." International Journal of Robotics and
            # Automation 18.1 (2003): 10-20.
            p = math.pi/4
            if dim == 1:
                weight = torch.tensor([[[0, p],
                                        [0, 0]]], dtype=dtype)
            if dim == 2:
                weight = torch.tensor([[[0, 0,  p,  p],
                                        [0, 0, -p, -p],
                                        [0, 0,  0,  0],
                                        [0, 0,  0,  0]]], dtype=dtype)
            if dim == 3:
                weight = torch.tensor(
                    [[[0, p, p, 0, p, 0, 0, 0],
                      [0, 0, 0, p, 0, p, 0, 0],
                      [0, 0, 0, p, 0, 0, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, p, p, 0],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, p],
                      [0, 0, 0, 0, 0, 0, 0, 0]]],
                    dtype=dtype)
            
            return weight.repeat(num_matrices,1,1)

        elif method == 'cayley':
            # The following matrices parametrize a Haar kernel matrix
            # when applying the Cayley transform. These can be found by
            # applying an inverse Cayley transform to a Haar kernel matrix.
            if dim == 1:
                p = -math.sqrt(2)/(2-math.sqrt(2))
                weight = torch.tensor([[[0, p],
                                        [0, 0]]], dtype=dtype)
            if dim == 2:
                p = .5
                weight = torch.tensor([[[0, 0,  p,  p],
                                        [0, 0, -p, -p],
                                        [0, 0,  0,  0],
                                        [0, 0,  0,  0]]], dtype=dtype)
            if dim == 3:
                p=1/math.sqrt(2)
                weight = torch.tensor(
                    [[[0, -p, -p,  0,  -p,   0,   0, 1-p],
                      [0,  0,  0, -p,   0,  -p, p-1,   0],
                      [0,  0,  0, -p,   0, p-1,  -p,   0],
                      [0,  0,  0,  0, 1-p,   0,   0,  -p],
                      [0,  0,  0,  0,   0,  -p,  -p,   0],
                      [0,  0,  0,  0,   0,   0,   0,  -p],
                      [0,  0,  0,  0,   0,   0,   0,  -p],
                      [0,  0,  0,  0,   0,   0,   0,   0]]],
                    dtype=dtype)
            return weight.repeat(num_matrices,1,1)

    if isinstance(init, str) and init in ['squeeze', 'pixel_shuffle', 'zeros']:
        if method == 'exp' or method == 'cayley':
            return torch.zeros(*kernel_matrix_shape, dtype=dtype)

    # An initialization of the weight can also be explicitly provided as a
    # numpy or torch tensor. If only one matrix is provided, this matrix
    # is copied num_matrices times.
    if not isinstance(init, str):
        try:
            init = torch.as_tensor(init, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError("init must be a supported name or an array-like matrix.") from exc
        if len(init.shape) == 2:
            init = init.reshape(1, *init.shape)
        if init.shape[0] == 1:
            init = init.repeat(num_matrices,1,1)
        if tuple(init.shape) != tuple(kernel_matrix_shape):
            raise ValueError(f"init must have shape {kernel_matrix_shape}.")
        return init

    else:
        raise ValueError(f"Unknown initialization: {init}.")


class OrthogonalChannelMixing(nn.Module):
    """Base class for all orthogonal channel mixing layers.

    """
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True,
                 **kwargs
                 ):
        super(OrthogonalChannelMixing, self).__init__()

        self.in_channels = in_channels
        initial = (torch.randn((in_channels, in_channels))
                   if method == 'householder'
                   else torch.zeros((in_channels, in_channels)))
        self.weight = nn.Parameter(initial, requires_grad=learnable)

        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if method not in ('exp', 'cayley', 'householder'):
            raise ValueError("method must be 'exp', 'cayley', or 'householder'.")
        if method == 'exp':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_exp__
        elif method == 'cayley':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_cayley__
        elif method == 'householder':
            self.__calculate_kernel_matrix__ \
                = __calculate_kernel_matrix_householder__

        self.kwargs = kwargs

    # Apply the chosen method to the weight in order to parametrize
    # an orthogonal matrix, then reshape into a convolutional kernel.
    @property
    def kernel_matrix(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return self.__calculate_kernel_matrix__(self.weight, **self.kwargs)

    @property
    def kernel_matrix_transposed(self):
        """The orthogonal matrix created by the chosen parametrisation method.
        """
        return torch.transpose(self.kernel_matrix, -1, -2)


class InvertibleChannelMixing1D(OrthogonalChannelMixing):
    """Orthogonal (and hence invertible) channel mixing layer for 1D data.

    This layer linearly combines the input channels to each output channel.
    Here, the number of output channels is the same as the number of input
    channels, and the matrix specifying the connectivity between the channels
    is orthogonal.

    :param in_channels:
        The number of input (and output) channels.
    :param method:
        The chosen method for parametrizing the orthogonal matrix which
        determines the orthogonal channel mixing. Either ``"exp"``, ``"cayley"``
        or ``"householder"``.

    """
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True,
                 **kwargs):
        super(InvertibleChannelMixing1D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable,
            **kwargs
        )
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1
        )

    def forward(self, x):
        return nn.functional.conv1d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose1d(x, self.kernel)

class InvertibleChannelMixing2D(OrthogonalChannelMixing):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True,
                 **kwargs
                 ):
        super(InvertibleChannelMixing2D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable,
            **kwargs
        )
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1, 1
        )

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose2d(x, self.kernel)

class InvertibleChannelMixing3D(OrthogonalChannelMixing):
    def __init__(self,
                 in_channels: int,
                 method: str = 'cayley',
                 learnable: bool = True,
                 **kwargs
                 ):
        super(InvertibleChannelMixing3D, self).__init__(
            in_channels=in_channels,
            method=method,
            learnable=learnable,
            **kwargs
        )
        self.kwargs = kwargs

    @property
    def kernel(self):
        return self.kernel_matrix.view(
            self.in_channels, self.in_channels, 1, 1, 1
        )

    def forward(self, x):
        return nn.functional.conv3d(x, self.kernel)

    def inverse(self, x):
        return nn.functional.conv_transpose3d(x, self.kernel)
