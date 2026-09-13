"""Inverse based gradient reconstruction for reversible module sequences.

Only segment boundary tensors are retained for backward. Intermediate states
are reconstructed one operation at a time, then differentiated locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.func import functional_call


def _tuple(value):
    return value if isinstance(value, tuple) else (value,)


class ReversibleModule(nn.Module):
    """Keep the old ``_fn`` checkpoint namespace without external dependencies."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        if not isinstance(fn, nn.Module) or not callable(getattr(fn, "inverse", None)):
            raise TypeError("Reversible modules must be nn.Module instances with inverse().")
        self._fn = fn

    def forward(self, *inputs):
        return self._fn(*inputs)

    def inverse(self, *outputs):
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
    def __init__(self, module: ReversibleModule, num_inputs: int, num_outputs: int,
                 reverse: bool = False):
        self.module = module
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.reverse = reverse
        self.call = _Call(module, reverse)
        self.inverse_call = _Call(module, not reverse)

    def inverted(self):
        return Step(self.module, self.num_outputs, self.num_inputs, not self.reverse)


@dataclass(frozen=True)
class BoundStep:
    step: Step
    parameter_indices: tuple[int, ...]
    parameter_names: tuple[str, ...]

    def _values(self, all_parameters):
        return tuple(all_parameters[i] for i in self.parameter_indices)

    def _mapping(self, all_parameters):
        return dict(zip(self.parameter_names, self._values(all_parameters)))

    def forward(self, inputs, all_parameters):
        return _tuple(functional_call(self.step.call, self._mapping(all_parameters), inputs))

    def inverse(self, outputs, all_parameters):
        return _tuple(functional_call(self.step.inverse_call,
                                      self._mapping(all_parameters), outputs))

    def vjp(self, inputs, output_gradients, all_parameters):
        values = self._values(all_parameters)

        def local(*arguments):
            state = arguments[:self.step.num_inputs]
            parameters = arguments[self.step.num_inputs:]
            mapping = dict(zip(self.parameter_names, parameters))
            return _tuple(functional_call(self.step.call, mapping, state))

        _, pullback = torch.func.vjp(local, *inputs, *values)
        return pullback(output_gradients)

    def jvp(self, inputs, input_tangents, all_parameters, parameter_tangents):
        values = self._values(all_parameters)
        tangents = tuple(parameter_tangents[i] for i in self.parameter_indices)

        def local(*arguments):
            state = arguments[:self.step.num_inputs]
            parameters = arguments[self.step.num_inputs:]
            mapping = dict(zip(self.parameter_names, parameters))
            return _tuple(functional_call(self.step.call, mapping, state))

        return torch.func.jvp(local, (*inputs, *values), (*input_tangents, *tangents))


@dataclass(frozen=True)
class Execution:
    steps: tuple[BoundStep, ...]
    num_inputs: int
    num_outputs: int
    num_parameters: int

    def run(self, state, parameters):
        for bound in self.steps:
            count = bound.step.num_inputs
            result = bound.forward(state[-count:], parameters)
            state = state[:-count] + result
        return state


class Plan:
    def __init__(self, steps: Sequence[Step], num_inputs: int):
        self.steps = tuple(steps)
        self.num_inputs = num_inputs
        count = num_inputs
        for step in steps:
            if count < step.num_inputs:
                raise ValueError("Invalid reversible sequence arity.")
            count += step.num_outputs - step.num_inputs
        self.num_outputs = count

    def inverted(self):
        return Plan(tuple(step.inverted() for step in reversed(self.steps)),
                    self.num_outputs)

    def bind(self):
        parameters = []
        indices = {}
        bound_steps = []
        for step in self.steps:
            local_indices = []
            local_names = []
            for name, parameter in step.module.named_parameters():
                if not parameter.requires_grad:
                    continue
                key = id(parameter)
                if key not in indices:
                    indices[key] = len(parameters)
                    parameters.append(parameter)
                local_indices.append(indices[key])
                local_names.append(f"module.{name}")
            bound_steps.append(BoundStep(step, tuple(local_indices), tuple(local_names)))
        return Execution(tuple(bound_steps), self.num_inputs, self.num_outputs,
                         len(parameters)), tuple(parameters)


class _Reconstruct(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(execution, *inputs_and_parameters):
        state = tuple(inputs_and_parameters[:execution.num_inputs])
        parameters = inputs_and_parameters[execution.num_inputs:]
        return execution.run(state, parameters)

    @staticmethod
    def setup_context(ctx, inputs, output):
        execution = inputs[0]
        tensors = inputs[1:]
        ctx.execution = execution
        ctx.save_for_backward(*_tuple(output), *tensors[execution.num_inputs:])

    @staticmethod
    def backward(ctx, *grad_outputs):
        execution = ctx.execution
        saved = ctx.saved_tensors
        state = tuple(saved[:execution.num_outputs])
        parameters = tuple(saved[execution.num_outputs:])
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
            for index, contribution in zip(bound.parameter_indices,
                                           local_gradients[input_count:]):
                previous = parameter_gradients[index]
                parameter_gradients[index] = (contribution if previous is None
                                              else previous + contribution)

        return (None, *gradients, *parameter_gradients)


class _ReconstructWithJVP(_Reconstruct):
    @staticmethod
    def setup_context(ctx, inputs, output):
        _Reconstruct.setup_context(ctx, inputs, output)
        ctx.save_for_forward(*inputs[1:])

    @staticmethod
    def jvp(ctx, _execution_tangent, *tangents):
        execution = ctx.execution
        saved = ctx.saved_tensors
        state = tuple(saved[:execution.num_inputs])
        parameters = tuple(saved[execution.num_inputs:])
        input_tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(state, tangents[:execution.num_inputs])
        )
        parameter_tangents = tuple(
            torch.zeros_like(value) if tangent is None else tangent
            for value, tangent in zip(parameters, tangents[execution.num_inputs:])
        )

        for bound in execution.steps:
            count = bound.step.num_inputs
            outputs, output_tangents = bound.jvp(
                state[-count:], input_tangents[-count:], parameters,
                parameter_tangents
            )
            state = state[:-count] + _tuple(outputs)
            input_tangents = input_tangents[:-count] + _tuple(output_tangents)

        return input_tangents


def apply_reconstruction(plan: Plan, *inputs):
    execution, parameters = plan.bind()
    function = _Reconstruct if torch.compiler.is_compiling() else _ReconstructWithJVP
    result = function.apply(execution, *inputs, *parameters)
    return result[0] if plan.num_outputs == 1 else result


def apply_autograd(plan: Plan, *inputs):
    execution, parameters = plan.bind()
    result = execution.run(tuple(inputs), parameters)
    return result[0] if plan.num_outputs == 1 else result
