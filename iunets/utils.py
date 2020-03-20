def get_num_channels(input_shape_or_channels):
    """
    Small helper function which outputs the number of
    channels regardless of whether the input shape or
    the number of channels were passed.
    """
    if hasattr(input_shape_or_channels,'__iter__'):
        return input_shape_or_channels[0]
    else:
        return input_shape_or_channels
    
    
def calculate_shapes_or_channels(
    input_shape_or_channels,
    slice_fraction,
    dim,
    i_level,
    sliced = False):
    
    # If input_shape_or_channels is the input shape
    if hasattr(input_shape_or_channels,'__iter__'):
        assert(len(input_shape_or_channels) == dim+1)
        if i_level == 0:
            return input_shape_or_channels
        else:
            # Copy to list to  prevent changing the original list.
            output_shape = [i for i in input_shape_or_channels] 

            initial_split = (input_shape_or_channels // slice_fraction + input_shape_or_channels%slice_fraction) * slice_fraction
            output_shape[0] = (initial_split * 2**(dim*i_level) // 
                (slice_fraction**i_level))
        if sliced:
            output_shape[0] = output_shape[0] // slice_fraction
            
            
        resolution_quotient = 2**i_level
        for j in range(1,len(output_shape)):
            output_shape[j] = output_shape[j] // resolution_quotient
        return output_shape
    
    else:
        # If input_shape_or_channels is just the number of channels
        if i_level == 0:
            return input_shape_or_channels
        else:
            initial_split = (input_shape_or_channels // slice_fraction + input_shape_or_channels%slice_fraction) * slice_fraction
            return (initial_split * 2**(dim*i_level) // (slice_fraction**i_level))
    
    
    
def calculate_shapes_or_channels_old(
    input_shape_or_channels,
    slice_fraction,
    dim,
    i_level,
    sliced = False):
    
    # If input_shape_or_channels is the input shape
    if hasattr(input_shape_or_channels,'__iter__'):
        assert(len(input_shape_or_channels) == dim+1)
        
        # Copy to list to  prevent changing the original list.
        output_shape = [i for i in input_shape_or_channels] 
        
        output_shape[0] = (output_shape[0] * 2**(dim*i_level) // 
            (slice_fraction**i_level))
        if sliced:
            output_shape[0] = output_shape[0] // slice_fraction
            
            
        resolution_quotient = 2**i_level
        for j in range(1,len(output_shape)):
            output_shape[j] = output_shape[j] // resolution_quotient
        return output_shape
    
    else:
        # If input_shape_or_channels is just the number of channels
        return (input_shape_or_channels * 2**(dim*i_level) // 
            (slice_fraction**i_level))