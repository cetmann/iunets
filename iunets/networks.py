import warnings

from typing import Union, Iterable, Callable, Any, Tuple

import numpy as np
import torch
from torch import nn
from memcnn import InvertibleModuleWrapper

from .layers import (create_standard_module, 
                     InvertibleDownsampling1D,
                     InvertibleDownsampling2D,
                     InvertibleDownsampling3D,
                     InvertibleUpsampling1D,
                     InvertibleUpsampling2D,
                     InvertibleUpsampling3D,
                     SplitChannels,
                     ConcatenateChannels)

from .utils import calculate_shapes_or_channels, get_num_channels


class iUNet(nn.Module):
    """Fully-invertible U-Net (iUNet).

    This model can be used for memory-efficient backpropagation, e.g. in
    high-dimensional (such as 3D) segmentation tasks.

    :param input_channels:
        The number of input channels, which is then also the number of output
        channel. Can also be the complete input shape (without batch dimension).
    :param architecture:
        Determines the number of invertible layers at each
        resolution (both left and right), e.g. ``(2,3,4)`` results in the
        following structure::
            2-----2
             3---3
              4-4

    :param dim: Either ``1``, ``2`` or ``3``, depending on the data. If
        ``None``, it is inferred from `input_channels`, if `input_channels`
        is the whole shape of the data.
    :param create_module_fn:
        Function which outputs an invertible layer. This layer
        should be a ``torch.nn.Module`` with a method ``forward(*x)``
        and a method ``inverse(*x)``. ``create_module_fn`` should have the
        signature ``create_module_fn(input_channels, **kwargs)``.
        Additional keyword arguments passed on via ``kwargs`` are
        ``dim`` (whether this is a 1D, 2D or 3D iUNet), the coordinates
        of the specific module within the iUNet (``LR``, ``level`` and
        ``module``) as well as ``architecture``.
    :param module_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to ``create_module_fn``.
    :param slice_mode:
        Controls the fraction of channels, which gets invertibly
        downsampled. Together with invertible downsampling
        Currently supported modes: ``"half"``, ``"constant"``.
        Defaults to ``"half"``.
    :param learnable_resampling:
        Whether to train the invertible learnable up- and downsampling
        or to leave it at the initialized values.
        Defaults to ``True``.
    :param resampling_stride:
        Controls the stride of the invertible up- and downsampling. Currently,
        only a stride of 2 across als dimensions is possible.
    :param resampling_method:
        Chooses the method for parametrizing orthogonal matrices for
        invertible up- and downsampling. Can be either ``"exp"`` (i.e.
        exponentiation of skew-symmetric matrices) or ``"cayley"`` (i.e.
        the Cayley transform, acting on skew-symmetric matrices).
        Defaults to ``"cayley"``.
    :param resampling_init:
        Sets the initialization for the learnable up- and downsampling
        operators. Can be ``"haar"``, ``"pixel_shuffle"`` (aliases:
        ``"squeeze"``, ``"zeros"``), a specific ``torch.Tensor`` or a
        ``numpy.ndarray``.
        Defaults to ``"haar"``, i.e. the `Haar transform`.
    :param resampling_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to the invertible up- and downsampling modules.
    :param disable_custom_gradient:
        If set to ``True``, `normal backpropagation` (i.e. storing
        activations instead of reconstructing activations) is used.
        Defaults to ``False``.
    :param padding_mode:
        If downsampling is not possible without residue
        (e.g. when halving spatial odd-valued resolutions), the
        input gets padded to allow for invertibility of the padded
        input. padding_mode takes the same keywords as
        ``torch.nn.functional.pad`` for ``mode``. If set to ``None``,
        this behavior is deactivated.
        Defaults to ``"constant"``.

    :param padding_value:
        If ``padding_mode`` is set to `constant`, this
        is the value that the input is padded with, e.g. 0.
        Defaults to ``0``.
    :param revert_input_padding:
        Whether to revert the input padding, if required. When using the
        iUNet for memory-efficient
        backpropagation, this can result in non-exact gradients.
        Defaults to ``False``.
    :param verbose:
        Level of verbosity. Currently only 0 (no warnings) or 1,
        which includes warnings.
        Defaults to ``1``.
    """
    def __init__(self,
                 input_channels: Union[int, Iterable[int]],
                 architecture: Tuple[int, ...],
                 dim: int = None,
                 create_module_fn: Callable[..., nn.Module] = create_standard_module,
                 module_kwargs: dict = None,
                 slice_mode: str = 'half',
                 learnable_resampling: bool = True,
                 resampling_stride: int = 2,
                 resampling_method: str = 'exp',
                 resampling_init: Union[str, np.ndarray, torch.Tensor] = 'haar',
                 resampling_kwargs: dict = None,
                 padding_mode: Union[str, type(None)] = 'constant',
                 padding_value: int = 0,
                 revert_input_padding: bool = False,
                 disable_custom_gradient: bool = False,
                 verbose: int = 1,
                 **kwargs: Any):

        super(iUNet, self).__init__()
        self.create_module_fn = create_module_fn

        if dim is None and not hasattr(input_channels, '__iter__'):
            raise AttributeError(
                "input_channels must be either the full shape  of the input "
                "(minus batch dimension) OR just the number  of channels, in "
                "which case dim has to be provided.")
        
        if hasattr(input_channels,'__len__'):
            dim = len(input_channels) - 1

        self.dim = dim
        self.architecture = architecture
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs

        # Standard behavior of self.slice_mode
        if slice_mode is "half":
            if self.dim == 1:
                raise AttributeError(
                    "slice_mode='half' not possible in 1D.")
            # The following results in a doubling of channels at each
            # downsampling stage for 2D or 3D data
            slice_fraction_dict = {2: 2,
                                   3: 4}
            slice_fraction = slice_fraction_dict[dim]
        if slice_mode is "constant":
            # The following results in a doubling of channels at each
            # downsampling stage for 2D or 3D data
            slice_fraction_dict = {1: 2,
                                   2: 4,
                                   3: 8}
            slice_fraction = slice_fraction_dict[dim]
        
        # Calculate the shapes of each level a priori
        self.shapes_or_channels = [calculate_shapes_or_channels(
            input_channels,
            slice_fraction,
            dim,
            i_level)
            for i_level in range(self.num_levels)]

        # Padding attributes
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.revert_input_padding = revert_input_padding

        assert(resampling_stride==2)
        self.resampling_stride = resampling_stride
        self.resampling_method = resampling_method
        self.resampling_init = resampling_init
        if resampling_kwargs is None:
            resampling_kwargs = {}
        self.resampling_kwargs = resampling_kwargs

        self.verbose = verbose

        # Create the architecture of the iUNet
        downsampling_op = [InvertibleDownsampling1D,
                           InvertibleDownsampling2D,
                           InvertibleDownsampling3D][dim-1]
        
        upsampling_op = [InvertibleUpsampling1D,
                         InvertibleUpsampling2D,
                         InvertibleUpsampling3D][dim-1]

        self.module_L = nn.ModuleList()
        self.module_R = nn.ModuleList()
        self.slice_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()

        for i, num_layers in enumerate(architecture):
            
            current_channels = get_num_channels(self.shapes_or_channels[i])
            if self.verbose and np.mod(current_channels, 2) == 1:
                warnings.warn(
                    "Odd number of channels detected. Expect faulty behaviour."
                )

            if i < len(architecture)-1:
                self.slice_layers.append(
                    InvertibleModuleWrapper(
                        SplitChannels(
                            current_channels
                            - current_channels // slice_fraction
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.concat_layers.append(
                    InvertibleModuleWrapper(
                        ConcatenateChannels(
                            current_channels
                            - current_channels // slice_fraction
                        ),
                        disable=disable_custom_gradient
                    )
                )

                downsampling = downsampling_op(
                    get_num_channels(
                        self.shapes_or_channels[i]
                    ) // slice_fraction,
                    stride=self.resampling_stride,
                    method=self.resampling_method,
                    init=self.resampling_init,
                    learnable=learnable_resampling,
                    **resampling_kwargs
                )

                upsampling = upsampling_op(
                    get_num_channels(
                        self.shapes_or_channels[i]
                    ) // slice_fraction * (2**dim),
                    stride=self.resampling_stride,
                    method=self.resampling_method,
                    init=self.resampling_init,
                    learnable=learnable_resampling,
                    **resampling_kwargs
                )
                
                # Initialize the learnable upsampling with the same
                # kernel as the learnable downsampling. This way, by
                # zero-initialization  of the coupling layers, the
                # invertible U-Net is initialized as the identity
                # function.
                if learnable_resampling:
                    upsampling.kernel_matrix.data = \
                        downsampling.kernel_matrix.data
                
                self.downsampling_layers.append(
                    InvertibleModuleWrapper(
                        downsampling,
                        disable=learnable_resampling
                    )
                )
                
                self.upsampling_layers.append(
                    InvertibleModuleWrapper(
                        upsampling,
                        disable=learnable_resampling
                    )
                )

            self.module_L.append(nn.ModuleList())
            self.module_R.append(nn.ModuleList())
            
            for j in range(num_layers):
                
                self.module_L[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.shapes_or_channels[i], 
                                 dim=self.dim,
                                 LR='R',
                                 level=i,
                                 module=j,
                                 architecture=self.architecture,
                                 **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )
                
                self.module_R[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.shapes_or_channels[i], 
                                 dim=self.dim,
                                 LR='R',
                                 level=i,
                                 module=j,
                                 architecture=self.architecture,
                                 **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )

    def get_padding(self, x):
        shape = x.shape[2:]
        f = 2 ** (len(self.architecture) - 1)
        padded_shape = [int(np.ceil(s / f)) * f for s in shape]
        total_padding = [p - s for (s, p) in zip(shape, padded_shape)]
        padding = [None] * (2 * len(shape))
        padding[::2] = [p // 2 for p in total_padding]
        padding[1::2] = [p - p // 2 for p in total_padding]
        return padded_shape, padding

    def padding_reversal(self, x, padding):
        if self.dim == 1:
            x = x[...,
                  padding[0]:-padding[1]]
        if self.dim == 2:
            x = x[...,
                  padding[0]:-padding[1],
                  padding[2]:-padding[3]]
        if self.dim == 3:
            x = x[...,
                  padding[0]:-padding[1],
                  padding[2]:-padding[3],
                  padding[4]:-padding[5]]
        return x

    def __check_stride_format__(self, stride):
        """Check whether the stride has the correct format to be parsed.
        """
        def raise_error():
            raise AttributeError(
                "resampling_stride has the wrong format."
            )
        if isinstance(stride, int):
            pass
        elif isinstance(stride, tuple):
            if len(stride) == self.dim:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_error()
        elif isinstance(stride, list):
            if len(stride) == self.num_levels-1:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_error()
        else:
            raise_error()

    def __format_stride__(self, stride):
        """Parses the resampling_stride.
        """
        self.__check_stride_format__(stride)
        if isinstance(stride, int):
            return [(stride,) * self.dim] * (self.num_levels - 1)
        if isinstance(stride, tuple):
            return [stride] * (self.num_levels - 1)
        if isinstance(stride, list):
            for i, element in enumerate(stride):
                if isinstance(element, int):
                    stride[i] = (element,) * self.dim
            return stride


    def forward(self, x):
        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn(
                    "Input resolution " + str(list(x.shape[2:])) +
                    " cannot be downsampled " + str(len(self.architecture)-1) +
                    " times without residuals. "
                    "Padding to resolution " + str(padded_shape) + " is " 
                    "applied with mode '" + self.padding_mode + "' to retain "
                    "invertibility. Set padding_mode=None to deactivate "
                    "padding. If so, expect errors."
                )
            x = nn.functional.pad(
                    x, padding, self.padding_mode, self.padding_value)

        # skip_inputs is a list of the skip connections
        skip_inputs = []

        # Left side
        for i in range(self.num_levels):
            depth = self.architecture[i]

            # RevNet L
            for j in range(depth):
                x = self.module_L[i][j](x)

            # Downsampling L
            if i < self.num_levels - 1:
                y, x = self.slice_layers[i](x)
                skip_inputs.append(y)
                x = self.downsampling_layers[i](x)

        # Right side
        for i in range(self.num_levels-1, -1, -1):
            depth = self.architecture[i]

            # Upsampling R
            if i < self.num_levels-1:
                y = skip_inputs.pop()
                x = self.upsampling_layers[i](x)
                x = self.concat_layers[i](y, x)

            # RevNet R
            for j in range(depth):
                x = self.module_R[i][j](x)

        if self.padding_mode is not None and self.revert_input_padding:
            if self.verbose:
                warnings.warn(
                    "revert_input_padding is set to True, which may yield "
                    "non-exact reconstructions of the unpadded input."
                )
            x = self.padding_reversal(x, padding)
        return x
    
    def inverse(self, x):

        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn(
                    "Input shape to the inverse mapping requires padding."
                )
            x = nn.functional.pad(
                    x, padding, self.padding_mode, self.padding_value)

        skip_inputs = []

        # Right side
        for i in range(self.num_levels):
            depth = self.architecture[i]

            # RevNet R
            for j in range(depth-1, -1, -1):
                x = self.module_R[i][j].inverse(x)

            # Downsampling R
            if i < self.num_levels - 1:
                y, x = self.concat_layers[i].inverse(x)
                skip_inputs.append(y)
                x = self.upsampling_layers[i].inverse(x)

        # Left side
        for i in range(self.num_levels-1, -1, -1):
            depth = self.architecture[i]

            # Upsampling L
            if i < self.num_levels-1:
                y = skip_inputs.pop()
                x = self.downsampling_layers[i].inverse(x)
                x = self.slice_layers[i].inverse(y, x)

            # RevNet L
            for j in range(depth-1, -1, -1):
                x = self.module_L[i][j].inverse(x)

        if self.padding_mode is not None and self.revert_input_padding:
            x = self.padding_reversal(x, padding)
        return x

