import torch
from torch.func import functional_call

from iunets import IUNet


def test_torch_func_transforms():
    model = IUNet((4,), (1,), 2, verbose=0)
    x = torch.randn(1, 4, 3, 3)
    objective = lambda value: model(value).square().sum()

    gradient = torch.func.grad(objective)(x)
    expected = torch.autograd.grad(objective(x.requires_grad_()), x)[0]
    torch.testing.assert_close(gradient, expected)

    tangent = torch.ones_like(x)
    _, directional = torch.func.jvp(model, (x,), (tangent,))
    _, objective_directional = torch.func.jvp(objective, (x,), (tangent,))
    torch.testing.assert_close(objective_directional, (gradient * tangent).sum(),
                               atol=1e-4, rtol=1e-4)
    assert directional.shape == x.shape
    assert torch.isfinite(directional).all()

    batched = torch.func.vmap(lambda sample: model(sample.unsqueeze(0)).squeeze(0))(
        x.expand(2, -1, -1, -1)
    )
    torch.testing.assert_close(batched[0], model(x)[0])
    assert torch.func.hessian(objective)(x).shape == x.shape + x.shape

    parameters = dict(model.named_parameters())
    param_grad = torch.func.grad(
        lambda values: functional_call(model, values, (x,)).square().sum()
    )(parameters)
    assert set(param_grad) == set(parameters)


def test_fullgraph_compiled_training():
    torch.manual_seed(17)
    model = IUNet((4, 8), (1, 1), 2, verbose=0)
    compiled = torch.compile(model, fullgraph=True, backend="aot_eager")
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    output = compiled(x)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())
