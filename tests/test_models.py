from collections import OrderedDict
import warnings

import pytest
import torch
from torch import nn

from iunets import IUNet, iUNet
from iunets.layers import (
    InvertibleDownsampling1D,
    InvertibleDownsampling2D,
    InvertibleDownsampling3D,
    InvertibleUpsampling1D,
)


SHAPES = [
    (1, (4, 6), (1, 4, 8)),
    (2, (4, 8), (1, 4, 4, 6)),
    (3, (4, 16), (1, 4, 4, 4, 4)),
]


@pytest.mark.parametrize("dim,channels,shape", SHAPES)
@pytest.mark.parametrize("method", ["cayley", "exp", "householder"])
def test_model_round_trip_and_gradient(dim, channels, shape, method):
    torch.manual_seed(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = IUNet(channels, (1, 1), dim, resampling_method=method,
                      channel_mixing_method=method, verbose=0)
    x = torch.randn(shape, requires_grad=True)
    y = model(x)
    torch.testing.assert_close(model.inverse(y), x, atol=2e-4, rtol=2e-4)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.requires_grad and p.grad is not None)


def test_padding_and_anisotropic_stride():
    model = IUNet((4, 8), (1, 1), 2, resampling_stride=(2, 1),
                  revert_input_padding=False, verbose=0)
    x = torch.randn(1, 4, 5, 7)
    assert model.get_padding(x) == ([6, 7], [0, 0, 0, 1])
    padded = model.pad(x)
    y = model(x)
    assert y.shape == padded.shape
    torch.testing.assert_close(model.inverse(y), padded, atol=1e-5, rtol=1e-5)


def test_single_level_and_layout(capsys):
    assert iUNet is IUNet
    model = IUNet((4,), (1,), 2, verbose=0)
    x = torch.randn(1, 4, 4, 4)
    torch.testing.assert_close(model.inverse(model(x)), x, atol=1e-5, rtol=1e-5)
    assert isinstance(model.layout(), str)
    model.print_layout()
    assert capsys.readouterr().out.strip() == model.layout().strip()


def test_one_dimensional_upsampling_and_validation():
    layer = InvertibleUpsampling1D(6, stride=2)
    x = torch.randn(1, 6, 4)
    torch.testing.assert_close(layer.inverse(layer(x)), x)
    for layer_cls, dim in ((InvertibleDownsampling1D, 1),
                           (InvertibleDownsampling2D, 2),
                           (InvertibleDownsampling3D, 3)):
        with pytest.raises(ValueError, match="stride"):
            layer_cls(2, stride=(0,) * dim)
    with pytest.raises(ValueError, match="divisible"):
        InvertibleUpsampling1D(3, stride=2)
    with pytest.raises(ValueError, match="gradient_mode"):
        IUNet((4,), (1,), 2, gradient_mode="unknown")
    with pytest.raises(TypeError, match="unexpected keyword"):
        IUNet((4,), (1,), 2, disable_custom_gradient=True)


class Scale(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, x):
        return x * self.scale

    def inverse(self, y):
        return y / self.scale


class Stochastic(Scale):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.dropout = nn.Dropout()

    def forward(self, x):
        return self.dropout(super().forward(x))


def test_custom_module_and_stateful_guidance():
    model = IUNet((4,), (2,), 2, create_module_fn=Scale, verbose=0)
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    torch.testing.assert_close(model.inverse(model(x)), x)
    model(x).sum().backward()
    assert x.grad is not None
    with pytest.raises(TypeError, match="inverse"):
        IUNet((4,), (1,), 2, create_module_fn=lambda *_a, **_k: nn.Identity())
    unsafe = IUNet((4,), (1,), 2, create_module_fn=Stochastic, verbose=0)
    with pytest.raises(RuntimeError, match="gradient_mode='autograd'"):
        unsafe(x)
    direct = IUNet((4,), (1,), 2, create_module_fn=Stochastic,
                   gradient_mode="autograd", verbose=0)
    direct(x)


def test_checkpoint_alias_migration():
    model = IUNet((4,), (1,), 2, verbose=0)
    state = model.state_dict()
    legacy = OrderedDict(state)
    for key, value in state.items():
        if ".F.seq." in key:
            legacy[key.replace(".F.seq.", ".F.F.")] = value.clone()
    restored = IUNet((4,), (1,), 2, verbose=0)
    restored.load_state_dict(legacy, strict=True)
    x = torch.randn(1, 4, 4, 4)
    torch.testing.assert_close(restored(x), model(x))


def test_gradient_modes_and_second_derivatives():
    torch.manual_seed(13)
    reconstructed = IUNet((4, 8), (1, 1), 2, verbose=0)
    direct = IUNet((4, 8), (1, 1), 2, gradient_mode="autograd", verbose=0)
    direct.load_state_dict(reconstructed.state_dict())
    x = torch.randn(1, 4, 4, 4)

    def derivatives(model):
        value = x.clone().requires_grad_()
        loss = model(value).square().sum()
        first = torch.autograd.grad(loss, (value, *model.parameters()),
                                    create_graph=True)
        second = torch.autograd.grad(first[0].square().sum(), value)[0]
        return loss, first, second

    a_loss, a_first, a_second = derivatives(reconstructed)
    b_loss, b_first, b_second = derivatives(direct)
    torch.testing.assert_close(a_loss, b_loss)
    for a, b in zip(a_first, b_first):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(a_second, b_second, atol=1e-2, rtol=1e-3)


def test_amp_cpu_training():
    model = IUNet((4,), (1,), 2, verbose=0)
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_saved_activation_storage_is_depth_independent():
    def saved_activation_elements(depth):
        model = IUNet((4,), (depth,), 2, module_kwargs={"depth": 1}, verbose=0)
        x = torch.randn(1, 4, 4, 4, requires_grad=True)
        saved = []

        def pack(tensor):
            if tensor.shape == x.shape:
                saved.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            loss = model(x).square().sum()
        loss.backward()
        return sum(saved)

    assert saved_activation_elements(5) == saved_activation_elements(1)
