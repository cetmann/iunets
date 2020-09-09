import warnings

from typing import (Union,
                    Iterable,
                    Callable)

import numpy as np
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
    """Fully invertible U-Net.
    """
    def __init__(self,
                 input_shape_or_channels : Union[int, Iterable[int]],
                 create_module_fn : Callable[..., nn.Module] = create_standard_module,
                 dim : Union[int, type(None)] = None,
                 architecture : Iterable[int, ...] = [2,2,2],
                 slice_fraction : Union[int, type(None)] = None,
                 learnable_downsampling : bool = True,
                 disable_custom_gradient : bool = False,
                 padding_mode : Union[str, type(None)] = 'constant',
                 padding_value : int = 0,
                 revert_input_padding : bool = False,
                 verbose : bool = True,
                 *args,
                 **kwargs):
        """

        Args:
            input_shape_or_channels : Either the number of channels
                or the full input shape (excluding the batch dimension).
            create_module_fn : Function which outputs an invertible layer.
            dim : Either 1D, 2D or 3D, depending on the data. If `None`,
                it is inferred from `input_shape_or_channels`, if
                `input_shape_or_channels` is the whole shape of the data.
            architecture : Determines the number of invertible layers
                at each resolution (both left and right). E.g.
                [2,3,4] leads to 2-----2.
                                  3---3
                                   4-4
            slice_fraction : The fraction of channels, which gets
                invertibly downsampled, such that e.g. for `2`, half
                of the channels are invertibly downsampled. This means
                that slicing and downsampling will altogether result in
                a doubling of channels. In 3D, `4` should be chosen
                for the same behavior.
            learnable_downsampling : Whether to train the invertible
                learnable up- and downsampling or to leave it at the
                initialized values.
            disable_custom_gradient : If set to True, normal
                backpropagation (i.e. with storing activations) is used.
            padding_mode : If downsampling is not possible without residue
                (e.g. when halving spatial odd-valued resolutions), the
                input gets padded to allow for invertibility of the padded
                input. padding_mode takes the same keywords as
                `torch.nn.functional.pad` for `mode`. If set to `None`,
                this behavior is deactivated.
            padding_value : If `padding_mode` is set to `constant`, this
                is the value that the input is padded with, e.g. 0.
            revert_input_padding : Whether to revert the input padding,
                if required. When using the iUNet for memory-efficient
                backpropagation, this can result in non-exact gradients.
            *args : Arbitrary-length iterable of values. These are passed
                on to create_module_fn.
            **kwargs : Arbitrary-length dictionary of key-value-pairs.
                These are passed on to create_module_fn.
        """
        super(iUNet, self).__init__()
        
        self.create_module_fn = create_module_fn
        
        if (dim is None and 
            not hasattr(input_shape_or_channels,'__iter__')):
            print(("input_shape_or_channels must be either the full shape " 
                  "of the input (minus batch dimension) OR just the number " 
                  "of channels, in which case dim has to be provided."))
        
        if hasattr(input_shape_or_channels,'__iter__'):
            dim = len(input_shape_or_channels) - 1
        
        
        self.dim = dim
        self.architecture = architecture
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)


        # Standard behavior of self.slice_fraction
        if slice_fraction is None:
            # The following results in a doubling of channels at each
            # downsampling stage for 2D or 3D data, and in constant
            # number of channels for 1D data.
            slice_fraction_dict = {1 : 2,
                                   2 : 2,
                                   3 : 4}
            slice_fraction = slice_fraction_dict[dim]
        self.slice_fraction = slice_fraction
        
        # Calculate the shapes of each level a priori
        self.shapes_or_channels = [calculate_shapes_or_channels(
            input_shape_or_channels,
            slice_fraction,
            dim,
            i_level) for i_level in range(self.num_levels)]

        # Padding attributes
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.revert_input_padding = revert_input_padding

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
        self.conc_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()

        for i, num_layers in enumerate(architecture):
            
            current_channels = get_num_channels(self.shapes_or_channels[i])
            if self.verbose:
                warnings.warn("Odd number of channels detected. Expect faulty "
                    "behaviour.")
            
            
            if i < len(architecture)-1:
                self.slice_layers.append(
                    InvertibleModuleWrapper(
                        SplitChannels(
                            current_channels
                            - current_channels // self.slice_fraction
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.conc_layers.append(
                    InvertibleModuleWrapper(
                        ConcatenateChannels(
                            current_channels
                            - current_channels // self.slice_fraction
                        ),
                        disable=disable_custom_gradient
                    )
                )

                downsampling = downsampling_op(
                    get_num_channels(
                        self.shapes_or_channels[i]
                    ) // slice_fraction,
                    learnable=learnable_downsampling
                )

                upsampling = upsampling_op(
                    get_num_channels(
                        self.shapes_or_channels[i]
                    ) // slice_fraction * (2**dim),
                    learnable=learnable_downsampling
                )
                
                # Initialize the learnabe upsampling with the same
                # kernel as the learnable downsampling. This way, by
                # zero-initialization  of the coupling layers, the
                # invertible U-Net is initialized as the identity
                # function.
                if learnable_downsampling:
                    upsampling.kernel_matrix.data = \
                        downsampling.kernel_matrix.data
                
                self.downsampling_layers.append(
                    InvertibleModuleWrapper(downsampling,
                        disable=disable_custom_gradient
                    )
                )
                
                self.upsampling_layers.append(
                    InvertibleModuleWrapper(upsampling,
                        disable=disable_custom_gradient
                    )
                )

            self.module_L.append(nn.ModuleList())
            self.module_R.append(nn.ModuleList())
            
            for j in range(num_layers):
                
                self.module_L[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.shapes_or_channels[i], 
                                 self.dim,
                                 'L', 
                                 i, 
                                 j, 
                                 self.architecture, 
                                 *args, 
                                 **kwargs),
                        disable=disable_custom_gradient
                    )
                )
                
                self.module_R[i].append(
                    InvertibleModuleWrapper(
                        create_module_fn(
                                 self.shapes_or_channels[i], 
                                 self.dim,
                                 'R', 
                                 i, 
                                 j, 
                                 self.architecture, 
                                 *args, 
                                 **kwargs),
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
            x = x[..., padding[0]:padding[1]]
        if self.dim == 2:
            x = x[..., padding[0]:padding[1],
                       padding[2]:padding[3]]
        if self.dim == 3:
            x = x[..., padding[0]:padding[1],
                       padding[2]:padding[3],
                       padding[4]:padding[5]]
        return x

    def forward(self, x):
        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warning("Input shape XXX cannot be downsampled YYY times without residuals. "
                    "Padding to shape ZZZ is applied with mode QQQ to retain invertibility. "
                    "Set padding=None to deactivate padding.")
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
                x = self.conc_layers[i](y, x)

            # RevNet R
            for j in range(depth):
                x = self.module_R[i][j](x)

        if self.padding is not None and self.revert_input_padding:
            if self.verbose:
                warnings.warning(
                    "revert_input_padding is set to True, which may yield "
                    "non-exact reconstructions of the unpadded input."
                )
            x = self.padding_reversal(x, padding)
        return x
    
    def inverse(self, x):

        padded_shape, padding = self.get_padding(x)
        if padded_shape != x.shape[2:] and self.padding_mode is not None:
            if self.verbose:
                warnings.warning("Input shape to the inverse mapping "
                    "requires padding.")
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
                y, x = self.conc_layers[i].inverse(x)
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

        if self.padding is not None and self.revert_input_padding:
            x = self.padding_reversal(x, padding)
        return x