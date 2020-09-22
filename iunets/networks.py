import warnings

from typing import Union, Iterable, Callable, Any, Tuple, Sized, List, Optional

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

from .utils import print_iunet_layout


class iUNet(nn.Module):
    """Fully-invertible U-Net (iUNet).

    This model can be used for memory-efficient backpropagation, e.g. in
    high-dimensional (such as 3D) segmentation tasks.

    :param in_channels:
        The number of input channels, which is then also the number of output
        channels. Can also be the complete input shape (without batch
        dimension).
    :param architecture:
        Determines the number of invertible layers at each
        resolution (both left and right), e.g. ``[2,3,4]`` results in the
        following structure::
            2-----2
             3---3
              4-4

    :param dim: Either ``1``, ``2`` or ``3``, signifying whether a 1D, 2D or 3D
        invertible U-Net should be created.
    :param create_module_fn:
        Function which outputs an invertible layer. This layer
        should be a ``torch.nn.Module`` with a method ``forward(*x)``
        and a method ``inverse(*x)``. ``create_module_fn`` should have the
        signature ``create_module_fn(in_channels, **kwargs)``.
        Additional keyword arguments passed on via ``kwargs`` are
        ``dim`` (whether this is a 1D, 2D or 3D iUNet), the coordinates
        of the specific module within the iUNet (``LR``, ``level`` and
        ``module_index``) as well as ``architecture``. By default, this creates
        an additive coupling layer, whose block consists of a number of
        convolutional layers, followed by a `leaky ReLU` activation function
        and an instance normalization layer. The number of blocks can be
        controlled by setting ``"block_depth"`` in ``module_kwargs``.
    :param module_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to ``create_module_fn``.
    :param slice_mode:
        Controls the fraction of channels, which gets invertibly
        downsampled. Together with invertible downsampling
        Currently supported modes: ``"double"``, ``"constant"``.
        Defaults to ``"double"``.
    :param learnable_resampling:
        Whether to train the invertible learnable up- and downsampling
        or to leave it at the initialized values.
        Defaults to ``True``.
    :param resampling_stride:
        Controls the stride of the invertible up- and downsampling.
        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        For example: ``2`` would result in a up-/downsampling with a factor of 2
        along each dimension; ``(2,1,4)`` would apply (at every
        resampling) a factor of 2, 1 and 4 for the height, width and depth
        dimensions respectively, whereas for a 3D iUNet with 3 up-/downsampling
        stages, ``[(2,1,3), (2,2,2), (4,3,1)]`` would result in different
        strides at different up-/downsampling stages.
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
        Whether to revert the input padding in the output, if desired.
        Defaults to ``True``.
    :param verbose:
        Level of verbosity. Currently only 0 (no warnings) or 1,
        which includes warnings.
        Defaults to ``1``.
    """
    def __init__(self,
                 in_channels: int,
                 architecture: Tuple[int, ...],
                 dim: int,
                 create_module_fn: Callable[[int, Optional[dict]], nn.Module]
                    = create_standard_module,
                 module_kwargs: dict = None,
                 slice_mode: str = "double",
                 learnable_resampling: bool = True,
                 resampling_stride: int = 2,
                 resampling_method: str = "cayley",
                 resampling_init: Union[str, np.ndarray, torch.Tensor] = "haar",
                 resampling_kwargs: dict = None,
                 padding_mode: Union[str, type(None)] = "constant",
                 padding_value: int = 0,
                 revert_input_padding: bool = True,
                 disable_custom_gradient: bool = False,
                 verbose: int = 1,
                 **kwargs: Any):

        super(iUNet, self).__init__()

        self.architecture = architecture
        self.dim = dim
        self.create_module_fn = create_module_fn
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs

        self.channels = [in_channels]
        self.channels_before_downsampling = []
        self.skipped_channels = []

        # --- Padding attributes ---
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.revert_input_padding = revert_input_padding

        # --- Invertible up- and downsampling attributes ---
        # Reformat resampling_stride to the standard format
        self.resampling_stride = self.__format_stride__(resampling_stride)
        # Calculate the channel multipliers per downsampling operation
        self.channel_multipliers = [
            int(np.prod(stride)) for stride in self.resampling_stride
        ]
        self.resampling_method = resampling_method
        self.resampling_init = resampling_init
        if resampling_kwargs is None:
            resampling_kwargs = {}
        self.resampling_kwargs = resampling_kwargs
        # Calculate the total downsampling factor per spatial dimension
        self.downsampling_factors = self.__total_downsampling_factor__(
            self.resampling_stride
        )

        # Standard behavior of self.slice_mode
        if slice_mode is "double" or slice_mode is "constant":
            if slice_mode is "double": factor = 2
            if slice_mode is "constant": factor = 1

            for i in range(len(architecture)-1):
                self.skipped_channels.append(
                    int(
                        max([1, np.floor(
                                (self.channels[i] *
                                 (self.channel_multipliers[i] - factor))
                                / self.channel_multipliers[i])]
                            )
                    )
                )
                self.channels_before_downsampling.append(
                        self.channels[i] - self.skipped_channels[-1]
                )
                self.channels.append(
                    self.channel_multipliers[i]
                    * self.channels_before_downsampling[i]
                )
        else:
            raise AttributeError(
                "Currently, only slice_mode='double' and 'constant' are "
                "supported."
            )

        # Verbosity level
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
            
            current_channels = self.channels[i]

            if i < len(architecture)-1:
                # Slice and concatenation layers
                self.slice_layers.append(
                    InvertibleModuleWrapper(
                        SplitChannels(
                            self.skipped_channels[i]
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.concat_layers.append(
                    InvertibleModuleWrapper(
                        ConcatenateChannels(
                            self.skipped_channels[i]
                        ),
                        disable=disable_custom_gradient
                    )
                )

                # Upsampling and downsampling layers
                downsampling = downsampling_op(
                    self.channels_before_downsampling[i],
                    stride=self.resampling_stride[i],
                    method=self.resampling_method,
                    init=self.resampling_init,
                    learnable=learnable_resampling,
                    **resampling_kwargs
                )

                upsampling = upsampling_op(
                    self.channels[i+1],
                    stride=self.resampling_stride[i],
                    method=self.resampling_method,
                    init=self.resampling_init,
                    learnable=learnable_resampling,
                    **resampling_kwargs
                )
                
                # Initialize the learnable upsampling with the same
                # kernel as the learnable downsampling. This way, by
                # zero-initialization of the coupling layers, the
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
                coordinate_kwargs = {
                    'dim': self.dim,
                    'LR': 'L',
                    'level': i,
                    'module_index': j,
                    'architecture': self.architecture,
                }
                self.module_L[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.channels[i],
                                 **coordinate_kwargs,
                                 **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )

                coordinate_kwargs['LR'] = 'R'
                self.module_R[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.channels[i],
                                 **coordinate_kwargs,
                                 **module_kwargs),
                        disable=disable_custom_gradient
                    )
                )


    def get_padding(self, x: torch.Tensor):
        """Calculates the required padding for the input.

        """
        shape = x.shape[2:]
        factors = self.downsampling_factors
        padded_shape = [
            int(np.ceil(s / f)) * f for (s,f) in zip(shape, factors)
        ]
        total_padding = [p - s for (s, p) in zip(shape, padded_shape)]

        # Pad evenly on all sides
        padding = [None] * (2 * len(shape))
        padding[::2] = [p - p // 2 for p in total_padding]
        padding[1::2] = [p // 2 for p in total_padding]

        # Weird thing about F.pad: While the torch data format is
        # (DHW), the padding format is (WHD).
        padding = padding[::-1]

        return padded_shape, padding

    def revert_padding(self, x: torch.Tensor, padding: List[int]):
        """Reverses a given padding.
        
        :param x:
            The image that was originally padded.
        :param padding:
            The padding that is removed from ``x``.
        """
        if self.dim == 1:
            x = x[:, :,
                  padding[0]:-padding[1]]
        if self.dim == 2:
            x = x[:, :,
                  padding[2]:-padding[3],
                  padding[0]:-padding[1]]
        if self.dim == 3:
            x = x[:, :,
                  padding[4]:-padding[5],
                  padding[2]:-padding[3],
                  padding[0]:-padding[1]]

        return x

    def __check_stride_format__(self, stride):
        """Check whether the stride has the correct format to be parsed.

        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        e.g. ``2`, ``(2,1,3)``, ``[(2,1,3), (2,2,2), (4,3,1)]``.
        """
        def raise_format_error():
            raise AttributeError(
                "resampling_stride has the wrong format. "
                "The format can be either a single integer, a single tuple "
                "(where the length corresponds to the spatial dimensions of the "
                "data), or a list containing either of the last two options "
                "(where the length of the list has to be equal to the number "
                "of downsampling operations), e.g. 2, (2,1,3), "
                "[(2,1,3), (2,2,2), (4,3,1)]. "
            )
        if isinstance(stride, int):
            pass
        elif isinstance(stride, tuple):
            if len(stride) == self.dim:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        elif isinstance(stride, list):
            if len(stride) == self.num_levels-1:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        else:
            raise_format_error()

    def __format_stride__(self, stride):
        """Parses the resampling_stride and reformats it into a standard format.

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

    def __total_downsampling_factor__(self, stride):
        factors = [1] * len(stride[0])
        for i, element_tuple in enumerate(stride):
            for j, element_int in enumerate(stride[i]):
                factors[j] = factors[j] * element_int
        return tuple(factors)

    def forward(self, x: torch.Tensor):
        """Applies the forward mapping of the iUNet to ``x``.
        """
        if not x.shape[1] == self.channels[0]:
            raise RuntimeError(
                "The number of channels does not match in_channels."
            )
        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warn(
                    "Input resolution {} cannot be downsampled {}  times "
                    "without residuals. Padding to resolution {} is  applied "
                    "with mode {} to retain invertibility. Set "
                    "padding_mode=None to deactivate padding. If so, expect "
                    "errors.".format(
                            list(x.shape[2:]),
                            len(self.architecture)-1,
                            padded_shape,
                            self.padding_mode
                        )
                    )

            x = nn.functional.pad(
                x, padding, self.padding_mode, self.padding_value
            )

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
            x = self.revert_padding(x, padding)
        return x
    
    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the iUNet to ``x``.
        """

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
            if self.verbose:
                warnings.warn(
                    "revert_input_padding is set to True, which may yield "
                    "non-exact reconstructions of the unpadded input."
                )
            x = self.revert_padding(x, padding)
        return x

    def print_layout(self):
        """Prints the layout of the iUNet.
        """
        print_iunet_layout(self)
