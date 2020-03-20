import warnings
import torch
from torch import nn
import memcnn
from memcnn import InvertibleModuleWrapper

from .layers import (create_standard_module, 
                     create_stable_1x1_module,
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
    def __init__(self,
                 input_shape_or_channels,
                 create_module_fn = create_standard_module,
                 dim = None,
                 architecture = [2,3,4],
                 slice_fraction = 2,
                 learnable_downsampling=True,
                 disable_custom_gradient=False,
                 *args,
                 **kwargs):
        super(iUNet, self).__init__()
        
        self.create_module_fn = create_module_fn
        
        if (dim is None and 
            not hasattr(input_shape_or_channels,'__iter__')):
            print(("input_shape_or_channels must be either the full shape " +
                  "of the input (minus batch dimension) OR just the number " +
                  "of channels, in which case dim has to be provided."))
        
        if hasattr(input_shape_or_channels,'__iter__'):
            dim = len(input_shape_or_channels) - 1
        
        
        self.dim = dim
        self.architecture = architecture
        self.disable_custom_gradient = disable_custom_gradient
        self.num_levels = len(architecture)
        self.slice_fraction = slice_fraction
        
        # Calculate the shapes of each level a priori
        self.shapes_or_channels = [calculate_shapes_or_channels(
            input_shape_or_channels,
            slice_fraction,
            dim,
            i_level) for i_level in range(self.num_levels)]
        
        
        # Create the layers of the iUNet
        
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
            warnings.warn("Odd number of channels detected. Expect faulty behaviour.")
            
            
            if i < len(architecture)-1:
                self.slice_layers.append(
                    InvertibleModuleWrapper(
                        SplitChannels(
                            current_channels - current_channels // self.slice_fraction 
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.conc_layers.append(
                    InvertibleModuleWrapper(
                        ConcatenateChannels(
                            current_channels - current_channels // self.slice_fraction
                        ),
                        disable=disable_custom_gradient
                    )
                )

                self.downsampling_layers.append(
                    InvertibleModuleWrapper(
                        downsampling_op(
                            get_num_channels(
                                self.shapes_or_channels[i]
                            ) // slice_fraction,
                        learnable=learnable_downsampling
                        ),
                        disable=disable_custom_gradient
                    )
                )
                self.upsampling_layers.append(
                    InvertibleModuleWrapper(
                        upsampling_op(
                            get_num_channels(
                                self.shapes_or_channels[i]
                            ) // slice_fraction * (2**dim),
                        learnable=learnable_downsampling
                        ),
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
        

    def forward(self, x):
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

        return x
    
    def inverse(self, x):
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

        return x