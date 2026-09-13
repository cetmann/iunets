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
    torch.testing.assert_close(
        objective_directional, (gradient * tangent).sum(), atol=1e-4, rtol=1e-4
    )
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
    reference = IUNet((4, 8), (1, 1), 2, gradient_mode="autograd", verbose=0)
    reference.load_state_dict(model.state_dict())
    compiled = torch.compile(model, fullgraph=True, backend="aot_eager")
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    output = compiled(x)
    output.square().sum().backward()
    other = x.detach().clone().requires_grad_()
    expected = reference(other)
    expected.square().sum().backward()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x.grad, other.grad, atol=1e-3, rtol=1e-3)
    for actual, direct in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, direct.grad, atol=1e-3, rtol=1e-3)
