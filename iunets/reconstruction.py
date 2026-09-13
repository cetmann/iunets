"""Inverse based gradient reconstruction for reversible module sequences.

Only segment boundary tensors are retained for backward. Intermediate states
are reconstructed one operation at a time, then differentiated locally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call

TORCH_23 = str(torch.__version__).startswith("2.3.")

# PyTorch 2.3 lazily imports Dynamo from inside functorch's custom Function
# dispatcher. Import it before entering a transform to avoid a circular
# initialization failure in that release.
if TORCH_23:
    import torch._dynamo


def _tuple(value):
    """Normalize a module result to a tuple for the execution planner."""

    return value if isinstance(value, tuple) else (value,)


class ReversibleModule(nn.Module):
    """Wrap an invertible module while preserving its checkpoint namespace.

    The wrapped module must implement ``inverse`` with the same input and
    output arity as ``forward`` in reverse order. Reconstruction mode calls
    both methods again during backward, so they should be deterministic and
    free of side effects.
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        if not isinstance(fn, nn.Module) or not callable(getattr(fn, "inverse", None)):
            raise TypeError(
                "Reversible modules must be nn.Module instances with inverse()."
            )
        self._fn = fn

    def forward(self, *inputs):
        """Delegate the forward call to the wrapped module."""

        return self._fn(*inputs)

    def inverse(self, *outputs):
        """Delegate the inverse call to the wrapped module."""

        return self._fn.inverse(*outputs)


class _Call(nn.Module):
    def __init__(self, module: ReversibleModule, reverse: bool):
        super().__init__()
        self.module = module
        self.reverse = reverse

    def forward(self, *values):
        if self.reverse:
            return self.module.inverse(*values)
        return self.module(*values)


class Step:
    """Describe one forward or inverse operation in a reversible plan."""

    def __init__(
        self,
        module: ReversibleModule,
        num_inputs: int,
        num_outputs: int,
        reverse: bool = False,
    ):
        self.module = module
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.reverse = reverse
        self.call = _Call(module, reverse)
        self.inverse_call = _Call(module, not reverse)

    def inverted(self):
        """Return a step that applies this operation in the opposite direction."""

        return Step(self.module, self.num_outputs, self.num_inputs, not self.reverse)


@dataclass(frozen=True)
class BoundStep:
    """A step with stable parameter indices for functional differentiation."""

    step: Step
    parameter_indices: tuple[int, ...]
    parameter_names: tuple[str, ...]

    def _values(self, all_parameters):
        return tuple(all_parameters[i] for i in self.parameter_indices)

    def _mapping(self, all_parameters):
        return dict(zip(self.parameter_names, self._values(all_parameters)))

    def forward(self, inputs, all_parameters):
        """Run this step with parameters supplied as a flat tuple."""

        return _tuple(
            functional_call(self.step.call, self._mapping(all_parameters), inputs)
        )

    def inverse(self, outputs, all_parameters):
        """Recover step inputs from outputs using the inverse module."""

        return _tuple(
            functional_call(
                self.step.inverse_call, self._mapping(all_parameters), outputs
            )
        )

    def vjp(self, inputs, output_gradients, all_parameters):
        """Compute local input and parameter gradients with ``torch.func.vjp``."""

        values = self._values(all_parameters)

        def local(*arguments):
            state = arguments[: self.step.num_inputs]
            parameters = arguments[self.step.num_inputs :]
            mapping = dict(zip(self.parameter_names, parameters))
            return _tuple(functional_call(self.step.call, mapping, state))

        _, pullback = torch.func.vjp(local, *inputs, *values)
        return pullback(output_gradients)

    def jvp(self, inputs, input_tangents, all_parameters, parameter_tangents):
        """Propagate input and parameter tangents through this step."""

        values = self._values(all_parameters)
        tangents = tuple(parameter_tangents[i] for i in self.parameter_indices)

        def local(*arguments):
            state = arguments[: self.step.num_inputs]
            parameters = arguments[self.step.num_inputs :]
            mapping = dict(zip(self.parameter_names, parameters))
            return _tuple(functional_call(self.step.call, mapping, state))

        return torch.func.jvp(local, (*inputs, *values), (*input_tangents, *tangents))


@dataclass(frozen=True)
class Execution:
    """Executable description of a sequence with explicit parameter inputs."""

    steps: tuple[BoundStep, ...]
    num_inputs: int
    num_outputs: int
    num_parameters: int

    def run(self, state, parameters):
        """Run the sequence using ``functional_call`` and explicit parameters."""

        for bound in self.steps:
            count = bound.step.num_inputs
            result = bound.forward(state[-count:], parameters)
            state = state[:-count] + result
        return state

    def run_compiled(self, state):
        """Run the sequence through direct module calls for compiler tracing.

        PyTorch 2.3 cannot trace ``functional_call`` inside a custom autograd
        function. The direct-call path is used only while compiling; eager
        reconstruction continues to use explicit functional parameters.
        """
        for bound in self.steps:
            count = bound.step.num_inputs
            result = _tuple(bound.step.call(*state[-count:]))
            state = state[:-count] + result
        return state


class Plan:
    """Validate and bind a reversible sequence for repeated model calls."""

    def __init__(self, steps: Sequence[Step], num_inputs: int):
        self.steps = tuple(steps)
        self.num_inputs = num_inputs
        count = num_inputs
        for step in steps:
            if count < step.num_inputs:
                raise ValueError("Invalid reversible sequence arity.")
            count += step.num_outputs - step.num_inputs
        self.num_outputs = count

        # Keep the call structure stable for torch.compile. Parameter values
        # are still read from modules on each invocation for functional_call.
        bound_steps = []
        parameter_count = 0
        for step in self.steps:
            names = tuple(
                f"module.{name}" for name, _ in step.module.named_parameters()
            )
            indices = tuple(range(parameter_count, parameter_count + len(names)))
            parameter_count += len(names)
            bound_steps.append(BoundStep(step, indices, names))
        self.execution = Execution(
            tuple(bound_steps), num_inputs, count, parameter_count
        )
        self.compiled_function = _make_compiled_function(self.execution)

    def inverted(self):
        """Return a plan that executes the sequence in reverse."""

        return Plan(
            tuple(step.inverted() for step in reversed(self.steps)), self.num_outputs
        )

    def bind(self):
        """Return the execution object and current module parameters."""

        parameters = tuple(
            parameter for step in self.steps for parameter in step.module.parameters()
        )
        return self.execution, parameters


class _Reconstruct(torch.autograd.Function):
    """Autograd bridge that retains only sequence boundaries for backward."""

    generate_vmap_rule = True

    @staticmethod
    def forward(execution, *inputs_and_parameters):
        state = tuple(inputs_and_parameters[: execution.num_inputs])
        parameters = inputs_and_parameters[execution.num_inputs :]
        return execution.run(state, parameters)

    @staticmethod
    def setup_context(ctx, inputs, output):
        execution = inputs[0]
        tensors = inputs[1:]
        ctx.execution = execution
        # Outputs are the final state; parameters are needed to replay each
        # inverse and local VJP. Intermediate states are deliberately omitted.
        ctx.save_for_backward(*_tuple(output), *tensors[execution.num_inputs :])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, *_reconstruct_backward(ctx, *grad_outputs))


def _reconstruct_backward(ctx, *grad_outputs):
    """Rebuild states from the output boundary and accumulate local VJPs.

    Only the final state of the sequence and its parameters are saved by the
    custom autograd function. Each step is inverted, differentiated locally,
    and then discarded before moving to the preceding step.
    """

    execution = ctx.execution
    saved = ctx.saved_tensors
    state = tuple(saved[: execution.num_outputs])
    parameters = tuple(saved[execution.num_outputs :])
    gradients = tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(state, grad_outputs)
    )
    parameter_gradients = [None] * execution.num_parameters

    for bound in reversed(execution.steps):
        count = bound.step.num_outputs
        inputs = bound.inverse(state[-count:], parameters)
        local_gradients = bound.vjp(inputs, gradients[-count:], parameters)
        input_count = bound.step.num_inputs
        gradients = gradients[:-count] + local_gradients[:input_count]
        state = state[:-count] + inputs
        for index, contribution in zip(
            bound.parameter_indices, local_gradients[input_count:]
        ):
            previous = parameter_gradients[index]
            parameter_gradients[index] = (
                contribution if previous is None else previous + contribution
            )

    return (*gradients, *parameter_gradients)


class _ReconstructWithJVP(_Reconstruct):
    """Reconstruction variant that also supports forward-mode AD transforms."""

    @staticmethod
    def setup_context(ctx, inputs, output):
        _Reconstruct.setup_context(ctx, inputs, output)
        ctx.save_for_forward(*inputs[1:])

    @staticmethod
    def jvp(ctx, _execution_tangent, *tangents):
        execution = ctx.execution
        saved = ctx.saved_tensors
        state = tuple(saved[: execution.num_inputs])
        parameters = tuple(saved[execution.num_inputs :])
        input_tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(state, tangents[: execution.num_inputs])
        )
        parameter_tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(parameters, tangents[execution.num_inputs :])
        )

        for bound in execution.steps:
            count = bound.step.num_inputs
            outputs, output_tangents = bound.jvp(
                state[-count:], input_tangents[-count:], parameters, parameter_tangents
            )
            state = state[:-count] + _tuple(outputs)
            input_tangents = input_tangents[:-count] + _tuple(output_tangents)

        return input_tangents


def _make_compiled_function(execution):
    """Capture the static plan before tracing; PyTorch 2.3 accepts tensors only."""

    class CompiledReconstruct(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs_and_parameters):
            state = tuple(inputs_and_parameters[: execution.num_inputs])
            parameters = inputs_and_parameters[execution.num_inputs :]
            output = execution.run_compiled(state)
            ctx.execution = execution
            ctx.save_for_backward(*output, *parameters)
            return output

        @staticmethod
        def backward(ctx, *grad_outputs):
            return _reconstruct_backward(ctx, *grad_outputs)

    return CompiledReconstruct


def apply_reconstruction(plan: Plan, *inputs):
    """Apply ``plan`` with bounded activation storage and inverse-based backward."""

    if torch.compiler.is_compiling() and TORCH_23:
        # The 2.3 compiler cannot trace module calls inside a custom
        # autograd Function. Keep fullgraph training available on that floor.
        result = plan.execution.run_compiled(tuple(inputs))
        return result[0] if plan.num_outputs == 1 else result
    execution, parameters = plan.bind()
    if torch.compiler.is_compiling():
        result = plan.compiled_function.apply(*inputs, *parameters)
    else:
        result = _ReconstructWithJVP.apply(execution, *inputs, *parameters)
    return result[0] if plan.num_outputs == 1 else result


def apply_autograd(plan: Plan, *inputs):
    """Apply ``plan`` with ordinary PyTorch autograd recording."""

    execution, parameters = plan.bind()
    result = execution.run(tuple(inputs), parameters)
    return result[0] if plan.num_outputs == 1 else result
