import torch

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


def eye_like(M, device=None, dtype=None):
    """Creates an identity matrix of the same shape as another matrix.

    For matrix M, the output is same shape as M, if M is a (n,n)-matrix.
    If M is a batch of m matrices (i.e. a (m,n,n)-tensor), create a batch of
    (n,n)-identity-matrices.

    Args:
        M (torch.Tensor) : A tensor of either shape (n,n) or (m,n,n), for
            which either an identity matrix or a batch of identity matrices
            of the same shape will be created.
        device (torch.device, optional) : The device on which the output
            will be placed. By default, it is placed on the same device
            as M.
        dtype (torch.dtype, optional) : The dtype of the output. By default,
            it is the same dtype as M.

    Returns:
        torch.Tensor : Identity matrix or batch of identity matrices.
    """
    assert(len(M.shape) in [2,3])
    assert(M.shape[-1] == M.shape[-2])
    n = M.shape[-1]
    if device is None:
        device = M.device
    if dtype is None:
        dtype = M.dtype
    eye = torch.eye(M.shape[-1], device=device, dtype=dtype)
    if len(M.shape)==2:
        return eye
    else:
        m = M.shape[0]
        return eye.view(-1, n, n).expand(m, -1, -1)
