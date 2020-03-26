import torch
from torch import nn
from .utils import get_num_channels
from .layers import StandardBlock

class StandardUNet(nn.Module):
    def __init__(self,
                input_shape_or_channels,
                dim=None,
                architecture=[2,2,2,2],
                base_filters=32,
                skip_connection=False,
                block_type=StandardBlock,
                zero_init=False,
                *args,
                **kwargs):
        super(StandardUNet, self).__init__()
        self.input_channels = get_num_channels(input_shape_or_channels)
        self.base_filters = base_filters
        self.architecture = architecture
        self.n_levels = len(self.architecture)
        self.dim = dim
        self.skip_connection = skip_connection
        self.block_type = block_type
        
        pool_ops = [nn.MaxPool1d, 
                    nn.MaxPool2d, 
                    nn.MaxPool3d]
        pool_op = pool_ops[dim-1]

        upsampling_ops = [nn.ConvTranspose1d,
                          nn.ConvTranspose2d,
                          nn.ConvTranspose3d]
        upsampling_op = upsampling_ops[dim-1]


        filters = self.base_filters
        filters_list = [filters]
        
        self.module_L = nn.ModuleList()
        self.module_R = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        
        # Left side of the U-Net
        for i in range(self.n_levels):
            self.module_L.append(nn.ModuleList())
            self.downsampling_layers.append(
                pool_op(kernel_size=2)
            )
            
            depth = architecture[i]    
                
            for j in range(depth):
                if i == 0 and j == 0:
                    in_channels = self.input_channels
                else:
                    in_channels = self.base_filters * (2**i)
                    
                if j == depth-1:
                    out_channels = self.base_filters * (2**(i+1))
                else:
                    out_channels = self.base_filters * (2**i)
                self.module_L[i].append(
                    self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs)
                )
                
        
        # Right side of the U-Net
        for i in range(self.n_levels-1):
            self.module_R.append(nn.ModuleList())
            depth = architecture[i]    
            for j in range(depth):
                if j == 0:
                    in_channels = 3*self.base_filters * (2**(i+1))
                else:
                    in_channels = self.base_filters * (2**(i+1))
                out_channels = self.base_filters * (2**(i+1))
                self.module_R[i].append(
                    self.block_type(self.dim, in_channels, out_channels, zero_init, *args, **kwargs)
                )
            

            self.upsampling_layers.append(
                upsampling_op(self.base_filters * (2**(i+2)),
                              self.base_filters * (2**(i+2)),
                              kernel_size=2,
                              stride=2)
            )
        
        if self.skip_connection:
            # We have to convert back to the original number of channels if
            # we want a skip connection. We do this with an appropriate
            # convolution.
            conv_ops = [nn.Conv1d,
                        nn.Conv2d,
                        nn.Conv3d]
            conv_op = conv_ops[self.dim-1]
            self.output_layer = conv_op(self.base_filters*2,
                                   self.input_channels,
                                   3,
                                   padding=1)


    def forward(self, input, *args, **kwargs):

        # FORWARD
        skip_inputs = []
        
        x = input
        
        # Left side
        for i in range(self.n_levels):
            depth = self.architecture[i]
            #  Left side
            for j in range(depth):
                x = self.module_L[i][j](x)

            # Downsampling L
            if i < self.n_levels - 1:
                skip_inputs.append(x)
                x = self.downsampling_layers[i](x)

        # Right side
        for i in range(self.n_levels - 2, -1, -1):
            depth = self.architecture[i]

            # Upsampling R
            x = self.upsampling_layers[i](x)
            y = skip_inputs.pop()
            x = torch.cat((x,y),dim=1)
            
            for j in range(depth):
                x = self.module_R[i][j](x)

        if self.skip_connection:
            x = self.output_layer(x) + input

        return x