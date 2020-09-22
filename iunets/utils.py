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
        
        
def print_iunet_layout(iunet):
    left = []
    right = []
    splits = []

    middle_padding = [''] * (iunet.num_levels)

    output = [''] * (iunet.num_levels)

    for i in range(iunet.num_levels):
        left.append(
            '-'.join([str(iunet.channels[i])] * iunet.architecture[i])
        )
        if i < iunet.num_levels-1:
            splits.append(
                '({}/{})'.format(
                    iunet.skipped_channels[i],
                    iunet.channels_before_downsampling[i]
                )
            )
        else:
            splits.append('')
        right.append(splits[-1] + '-' + left[-1])
        left[-1] = left[-1] + '-'+ splits[-1]


    for i in range(iunet.num_levels - 1, -1, -1):
        if i < iunet.num_levels-1:
            middle_padding[i] = \
                ''.join(['-'] * max([len(output[i+1]) - len(splits[i]),4]))
        output[i] = left[i] + middle_padding[i] + right[i]

    for i in range(iunet.num_levels):
        if i>0:
            outside_padding = len(output[0]) - len(output[i]) 
            _left =  outside_padding // 2
            left_padding = ''.join(['-'] * _left)
            _right = outside_padding - _left
            right_padding = ''.join(['-'] * _right)
            output[i] = ''.join([left_padding, output[i], right_padding])
        print(output[i])