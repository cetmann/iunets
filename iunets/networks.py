from __future__ import annotations

import math
import warnings
from collections import OrderedDict

from typing import Union, Iterable, Callable, Any, Tuple, Sized, List, Optional

import torch
from torch import nn

from .reconstruction import (Plan, ReversibleModule, Step, apply_autograd,
                             apply_reconstruction)

from .layers import (create_standard_module, 
                     InvertibleDownsampling1D,
                     InvertibleDownsampling2D,
                     InvertibleDownsampling3D,
                     InvertibleUpsampling1D,
                     InvertibleUpsampling2D,
                     InvertibleUpsampling3D,
                     InvertibleChannelMixing1D,
                     InvertibleChannelMixing2D,
                     InvertibleChannelMixing3D,
                     SplitChannels,
                     ConcatenateChannels)

from .utils import print_iunet_layout

CreateModuleFnType = Callable[..., nn.Module]

class IUNet(nn.Module):
    """Fully-invertible U-Net (iUNet).

    This model can be used for memory-efficient backpropagation, e.g. in
    high-dimensional (such as 3D) segmentation tasks.

    :param channels:
        The number of channels at each resolution. For example: If one wants
        5 resolution levels (i.e. 3 up-/downsampling operations), it should be
        a tuple of 4 numbers, e.g. ``(32,64,128,256,384)``.
    :param architecture:
        Determines the number of invertible layers at each
        resolution (both left and right), e.g. ``[2,3,4]`` results in the
        following structure::
            2-----2
             3---3
              4-4
        Must be the same length as ``channels``.
    :param dim: Either ``1``, ``2`` or ``3``, signifying whether a 1D, 2D or 3D
        invertible U-Net should be created.
    :param create_module_fn:
        Function which outputs an invertible layer. This layer
        should be a ``torch.nn.Module`` with a method ``forward(*x)``
        and a method ``inverse(*x)``. ``create_module_fn`` should have the
        signature ``create_module_fn(in_channels, **kwargs)``.
        Additional keyword arguments passed on via ``kwargs`` are
        ``dim`` (whether this is a 1D, 2D or 3D iUNet), the coordinates
        of the specific module within the iUNet (``branch``, ``level`` and
        ``module_index``) as well as ``architecture``. By default, this creates
        an additive coupling layer, whose block consists of a number of
        convolutional layers, followed by an instance normalization layer and
        a `leaky ReLU` activation function. The number of blocks can be
        controlled by setting ``"depth"`` in ``module_kwargs``, whose default
        value is ``2``.
    :param module_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to ``create_module_fn``.
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
    :param channel_mixing_freq:
        How often an `invertible channel mixing` is applied, which is (in 2D)
        is an orthogonal 1x1-convolution. ``-1`` means that this will only be
        applied before the channel splitting and before the recombination in the
        decoder branch. For any other ``n``, this means that every ``n``-th
        module is followed by an invertible channel mixing. In particular,``0``
        deactivates the usage of invertible channel mixing.
        Defaults to ``-1``.
    :param channel_mixing_method:
        How the orthogonal matrix for invertible channel mixing is parametrized.
        Same has ``resampling_method``.
        Defaults to ``"cayley"``.
    :param channel_mixing_kwargs:
        ``dict`` of optional, additional keyword arguments that are
        passed on to the invertible channel mixing modules.
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
        Whether to revert the input padding in the output, such that the
        input resolution is preserved, even when padding is required.
        Defaults to ``True``.
    :param gradient_mode:
        ``"reconstruct"`` uses inverse based gradient reconstruction;
        ``"autograd"`` uses ordinary PyTorch backpropagation. Custom
        reversible modules must be deterministic and side effect free in
        reconstruction mode. Defaults to ``"reconstruct"``.
    :param verbose:
        Level of verbosity. Currently only 0 (no warnings) or 1,
        which includes warnings.
        Defaults to ``1``.
    """
    def __init__(self,
                 channels: Tuple[int, ...],
                 architecture: Tuple[int, ...],
                 dim: int,
                 create_module_fn: CreateModuleFnType
                    = create_standard_module,
                 module_kwargs: dict = None,
                 learnable_resampling: bool = True,
                 resampling_stride: int = 2,
                 resampling_method: str = "cayley",
                 resampling_init: Any = "haar",
                 resampling_kwargs: dict = None,
                 learnable_channel_mixing: bool = True,
                 channel_mixing_freq: int = -1,
                 channel_mixing_method: str = 'cayley',
                 channel_mixing_kwargs: dict = None,
                 padding_mode: Union[str, type(None)] = "constant",
                 padding_value: int = 0,
                 revert_input_padding: bool = True,
                 gradient_mode: str = "reconstruct",
                 verbose: int = 1):

        super().__init__()

        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3.")
        if (not isinstance(channels, (tuple, list)) or
                not isinstance(architecture, (tuple, list)) or
                not architecture or len(channels) != len(architecture)):
            raise ValueError("channels and architecture must have the same nonzero length.")
        if any(type(value) is not int or value <= 0 for value in channels):
            raise ValueError("channels must contain positive integers.")
        if any(type(value) is not int or value < 0 for value in architecture):
            raise ValueError("architecture must contain nonnegative integers.")
        if gradient_mode not in ("reconstruct", "autograd"):
            raise ValueError("gradient_mode must be 'reconstruct' or 'autograd'.")
        if type(channel_mixing_freq) is not int or channel_mixing_freq < -1:
            raise ValueError("channel_mixing_freq must be -1 or a nonnegative integer.")
        if not callable(create_module_fn):
            raise TypeError("create_module_fn must be callable.")
        for name, value in (("module_kwargs", module_kwargs),
                            ("resampling_kwargs", resampling_kwargs),
                            ("channel_mixing_kwargs", channel_mixing_kwargs)):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{name} must be a dictionary.")
        if padding_mode not in (None, "constant", "reflect", "replicate", "circular"):
            raise ValueError("padding_mode must be a supported torch.nn.functional.pad mode or None.")

        self.architecture = architecture
        self.dim = dim
        self.create_module_fn = create_module_fn
        self.gradient_mode = gradient_mode
        self.verbose = verbose
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs

        self.channels = [channels[0]]
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
            math.prod(stride) for stride in self.resampling_stride
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

        # --- Invertible channel mixing attributes ---
        self.learnable_channel_mixing = learnable_channel_mixing
        self.channel_mixing_freq = channel_mixing_freq
        self.channel_mixing_method = channel_mixing_method
        if channel_mixing_kwargs is None:
            channel_mixing_kwargs = {}
        self.channel_mixing_kwargs = channel_mixing_kwargs

        # --- Setting up the required channel numbers ---
        # The user-specified channel numbers can potentially not be enforced.
        # Hence, we choose the best possible approximation.
        desired_channels = channels
        channel_errors = [] # Measure how far we are off.

        for i in range(len(architecture)-1):
            factor = desired_channels[i+1] / self.channels[i]
            skip_fraction = (self.channel_multipliers[i] - factor) \
                               / self.channel_multipliers[i]
            self.skipped_channels.append(
                    max(1, round(self.channels[i] * skip_fraction))
            )
            self.channels_before_downsampling.append(
                    self.channels[i] - self.skipped_channels[-1]
            )
            self.channels.append(
                self.channel_multipliers[i]
                * self.channels_before_downsampling[i]
            )
            if self.channels_before_downsampling[i] <= 0:
                raise ValueError(
                    "Each downsampling level must retain at least one channel; "
                    "increase channels or adjust the requested channel schedule."
                )

            channel_errors.append(
                abs(self.channels[i+1] - desired_channels[i+1])
                / desired_channels[i+1]
            )

        # Convert to tuples (standard convention)
        self.channels = tuple(
            self.channels
        )
        self.channels_before_downsampling = tuple(
            self.channels_before_downsampling
        )
        self.skipped_channels = tuple(
            self.skipped_channels
        )

        if list(channels) != list(self.channels) and self.verbose:
            warnings.warn(
                "Could not exactly create an iUNet with channels={} and "
                "resampling_stride={}. Instead using closest achievable "
                "configuration: channels={}. Average relative error: {}".format(
                    channels,
                    self.resampling_stride,
                    self.channels,
                    sum(channel_errors) / len(channel_errors)
                )
            )

        # --- Create the architecture of the iUNet ---
        downsampling_op = [InvertibleDownsampling1D,
                           InvertibleDownsampling2D,
                           InvertibleDownsampling3D][dim-1]
        
        upsampling_op = [InvertibleUpsampling1D,
                         InvertibleUpsampling2D,
                         InvertibleUpsampling3D][dim-1]

        channel_mixing_op = [InvertibleChannelMixing1D,
                             InvertibleChannelMixing2D,
                             InvertibleChannelMixing3D][dim-1]

        self.encoder_modules = nn.ModuleList()
        self.decoder_modules = nn.ModuleList()
        self.slice_layers = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()

        for i, num_layers in enumerate(architecture):
            
            if i < len(architecture)-1:
                # Slice and concatenation layers
                self.slice_layers.append(
                    ReversibleModule(SplitChannels(self.skipped_channels[i]))
                )
                self.concat_layers.append(
                    ReversibleModule(ConcatenateChannels(self.skipped_channels[i]))
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
                
                # Paired operators must start with the same orthogonal map.
                with torch.no_grad():
                    upsampling.weight.copy_(downsampling.weight)
                
                self.downsampling_layers.append(
                    ReversibleModule(downsampling)
                )
                
                self.upsampling_layers.append(
                    ReversibleModule(upsampling)
                )

            self.encoder_modules.append(nn.ModuleList())
            self.decoder_modules.append(nn.ModuleList())

            # Shorthand for adding channel mixing layers, which all have the
            # same specifications.
            def add_channel_mixing(obj, module_list):
                module_list.append(
                    ReversibleModule(
                        channel_mixing_op(
                            in_channels=obj.channels[i],
                            method=obj.channel_mixing_method,
                            learnable=obj.learnable_channel_mixing,
                            **obj.channel_mixing_kwargs)
                    )
                )

            for j in range(num_layers):
                # create_module_fn is additionally supplied with a dictionary
                # containing information about the whereabouts of the current
                # module.
                coordinate_kwargs = {
                    'dim': self.dim,
                    'branch': 'encoder',
                    'level': i,
                    'module_index': j,
                    'architecture': self.architecture,
                }
                self.encoder_modules[i].append(
                    ReversibleModule(
                        create_module_fn(
                                 self.channels[i],
                                 **coordinate_kwargs,
                                 **module_kwargs)
                    )
                )

                coordinate_kwargs['branch'] = 'decoder'
                self.decoder_modules[i].append(
                    ReversibleModule(
                        create_module_fn(
                                 self.channels[i],
                                 **coordinate_kwargs,
                                 **module_kwargs)
                    )
                )

                # Rules for channel mixing
                if self.channel_mixing_freq == -1 and i!=len(architecture)-1:
                    if j == 0:
                        add_channel_mixing(self, self.decoder_modules[i])
                    if j == num_layers - 1:
                        add_channel_mixing(self, self.encoder_modules[i])

                if (self.channel_mixing_freq > 0 and
                    j % self.channel_mixing_freq == self.channel_mixing_freq-1):
                    add_channel_mixing(self, self.encoder_modules[i])
                    add_channel_mixing(self, self.decoder_modules[i])

        encode_steps = []
        for i in range(self.num_levels):
            encode_steps.extend(Step(module, 1, 1)
                                for module in self.encoder_modules[i])
            if i < self.num_levels - 1:
                encode_steps.append(Step(self.slice_layers[i], 1, 2))
                encode_steps.append(Step(self.downsampling_layers[i], 1, 1))

        decode_steps = []
        for i in range(self.num_levels - 1, -1, -1):
            if i < self.num_levels - 1:
                decode_steps.append(Step(self.upsampling_layers[i], 1, 1))
                decode_steps.append(Step(self.concat_layers[i], 2, 1))
            decode_steps.extend(Step(module, 1, 1)
                                for module in self.decoder_modules[i])

        self._encode_plan = Plan(encode_steps, 1)
        self._decode_plan = Plan(decode_steps, self.num_levels)
        self._decoder_inverse_plan = self._decode_plan.inverted()
        self._encoder_inverse_plan = self._encode_plan.inverted()
        self._has_stateful_modules = any(
            isinstance(module, (nn.modules.batchnorm._BatchNorm,
                                nn.modules.instancenorm._InstanceNorm, nn.Dropout,
                                nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))
            and (not isinstance(module, (nn.modules.batchnorm._BatchNorm,
                                        nn.modules.instancenorm._InstanceNorm))
                 or module.track_running_stats)
            for module in self.modules()
        )

    def get_padding(self, x: torch.Tensor):
        """Calculates the required padding for the input.

        """
        shape = x.shape[2:]
        factors = self.downsampling_factors
        padded_shape = [
            ((s + f - 1) // f) * f for (s, f) in zip(shape, factors)
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
        shape = x.shape
        if self.dim == 1:
            x = x[:, :,
                  padding[0]:shape[2]-padding[1]]
        if self.dim == 2:
            x = x[:, :,
                  padding[2]:shape[2]-padding[3],
                  padding[0]:shape[3]-padding[1]]
        if self.dim == 3:
            x = x[:, :,
                  padding[4]:shape[2]-padding[5],
                  padding[2]:shape[3]-padding[3],
                  padding[0]:shape[4]-padding[1]]

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
            raise ValueError(
                "resampling_stride has the wrong format. "
                "The format can be either a single integer, a single tuple "
                "(where the length corresponds to the spatial dimensions of the "
                "data), or a list containing either of the last two options "
                "(where the length of the list has to be equal to the number "
                "of downsampling operations), e.g. 2, (2,1,3), "
                "[(2,1,3), (2,2,2), (4,3,1)]. "
            )
        if type(stride) is int:
            if stride <= 0:
                raise_format_error()
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
        if self.num_levels == 1:
            return []
        self.__check_stride_format__(stride)
        if isinstance(stride, int):
            return [(stride,) * self.dim] * (self.num_levels - 1)
        if isinstance(stride, tuple):
            return [stride] * (self.num_levels - 1)
        if isinstance(stride, list):
            return [((element,) * self.dim if isinstance(element, int)
                     else element) for element in stride]

    def __total_downsampling_factor__(self, stride):
        """Calculates the total downsampling factor per spatial dimension.
        """
        factors = [1] * self.dim
        for i, element_tuple in enumerate(stride):
            for j, element_int in enumerate(stride[i]):
                factors[j] = factors[j] * element_int
        return tuple(factors)

    def pad(self, x, padded_shape=None, padding=None):
        """Applies the chosen padding to the input, if required.
        """
        if self.padding_mode is None:
            raise AttributeError(
                "padding_mode in {} is set to None.".format(self)
            )

        if padded_shape is None or padding is None:
            padded_shape, padding = self.get_padding(x)

        if tuple(padded_shape) != tuple(x.shape[2:]):
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

            if self.padding_mode == "constant":
                x = nn.functional.pad(x, padding, mode="constant",
                                      value=self.padding_value)
            else:
                x = nn.functional.pad(x, padding, mode=self.padding_mode)
        return x

    def _apply_plan(self, plan: Plan, *inputs):
        if self.gradient_mode == "reconstruct" and self.training and self._has_stateful_modules:
            raise RuntimeError(
                "Reconstruction requires deterministic, side effect free modules in "
                "training mode. Use gradient_mode='autograd' for dropout or "
                "BatchNorm with running statistics."
            )
        if self.gradient_mode == "autograd" or not torch.is_grad_enabled() or not plan.steps:
            return apply_autograd(plan, *inputs)
        return apply_reconstruction(plan, *inputs)

    def encode(self, x, use_padding=False):
        """Encodes x, i.e. applies the contractive part of the iUNet.
        """
        if use_padding:
            x = self.pad(x)
        return self._apply_plan(self._encode_plan, x)

    def decode(self, *codes):
        """Applies the expansive, i.e. decoding, portion of the iUNet.
        """
        if len(codes) != self.num_levels:
            raise ValueError(f"decode expects {self.num_levels} code tensors.")
        return self._apply_plan(self._decode_plan, *codes)

    def forward(self, x: torch.Tensor):
        """Applies the forward mapping of the iUNet to ``x``.
        """
        if x.ndim != self.dim + 2:
            raise ValueError(f"Expected a {self.dim + 2}D tensor (batch, channels, spatial dimensions).")
        if not x.shape[1] == self.channels[0]:
            raise ValueError(
                f"Expected {self.channels[0]} input channels, got {x.shape[1]}."
            )
        if self.padding_mode is not None:
            padded_shape, padding = self.get_padding(x)
            x = self.pad(x, padded_shape, padding)

        code = self.encode(x, use_padding=False)
        x = self.decode(*(code if isinstance(code, tuple) else (code,)))

        if self.padding_mode is not None and self.revert_input_padding:
            x = self.revert_padding(x, padding)
        return x

    def decoder_inverse(self, x, use_padding=False):
        """Applies the inverse of the decoder portion of the iUNet.
        """
        if use_padding:
            x = self.pad(x)
        return self._apply_plan(self._decoder_inverse_plan, x)

    def encoder_inverse(self, *codes):
        """Applies the inverse of the encoder portion of the iUNet.
        """
        if len(codes) != self.num_levels:
            raise ValueError(f"encoder_inverse expects {self.num_levels} code tensors.")
        return self._apply_plan(self._encoder_inverse_plan, *codes)
    
    def inverse(self, x: torch.Tensor):
        """Applies the inverse of the iUNet to ``x``.
        """
        if x.ndim != self.dim + 2:
            raise ValueError(f"Expected a {self.dim + 2}D tensor (batch, channels, spatial dimensions).")
        if not x.shape[1] == self.channels[0]:
            raise ValueError(
                f"Expected {self.channels[0]} input channels, got {x.shape[1]}."
            )
        if self.padding_mode is not None:
            padded_shape, padding = self.get_padding(x)
            x = self.pad(x, padded_shape, padding)

        code = self.decoder_inverse(x, use_padding=False)
        x = self.encoder_inverse(*(code if isinstance(code, tuple) else (code,)))

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
        print(self.layout())

    def layout(self) -> str:
        """Return a compact text diagram of channel flow through the model."""
        return print_iunet_layout(self)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load current and 0.1 checkpoints with duplicated block aliases."""
        migrated = OrderedDict(state_dict)
        if hasattr(state_dict, "_metadata"):
            migrated._metadata = state_dict._metadata
        for key in tuple(migrated):
            if ".F.F." in key:
                replacement = key.replace(".F.F.", ".F.seq.")
                migrated.setdefault(replacement, migrated[key])
                del migrated[key]
        return super().load_state_dict(migrated, strict=strict, assign=assign)


iUNet = IUNet
