from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the input values.

        Args:
        ----
            vals (ScalarLike): The input values.

        Returns:
        -------
            Scalar: The result of the scalar function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to a and b.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the natural logarithm.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the natural logarithm.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of a * b.

        """
        # Save the input values for the backward pass
        ctx.save_for_backward(a, b)

        # Compute and return the product
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to a and b.

        """
        # Retrieve the input values from the context
        a, b = ctx.saved_values

        # Compute the gradients
        # For f(x, y) = x * y, df/dx = y and df/dy = x
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the inverse function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of 1/a.

        """
        # Save the input value for the backward pass
        ctx.save_for_backward(a)

        # Compute and return the inverse
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the inverse function.

        Args:
        ----
            ctx (Context): The context from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # Retrieve the input value from the context
        (a,) = ctx.saved_values

        # Compute the gradient
        # For f(x) = 1/x, df/dx = -1/x^2
        return -1.0 / (a * a) * d_output


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the negation function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The negation of the input (-a).

        """
        # No need to save any values for backward pass
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the negation function.

        Args:
        ----
            ctx (Context): The context from the forward pass (unused in this case).
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # For f(x) = -x, df/dx = -1
        # The gradient is always -1 times the output gradient
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The sigmoid of the input, 1 / (1 + e^(-a)).

        """
        # Compute the sigmoid
        sigmoid_value = 1.0 / (1.0 + operators.exp(-a))

        # Save the sigmoid value for the backward pass
        ctx.save_for_backward(sigmoid_value)

        return sigmoid_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # Retrieve the sigmoid value saved during the forward pass
        sigmoid_value = ctx.saved_values[0]

        # For sigmoid function, f'(x) = f(x) * (1 - f(x))
        # Where f(x) is the sigmoid value
        return d_output * sigmoid_value * (1 - sigmoid_value)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The ReLU of the input, max(0, a).

        """
        # Compute the ReLU
        relu_value = max(0.0, a)

        # Save the input value for the backward pass
        ctx.save_for_backward(a)

        return relu_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # Retrieve the input value saved during the forward pass
        a = ctx.saved_values[0]

        # For ReLU function, f'(x) = 1 if x > 0, else 0
        return d_output if a > 0 else 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The exponential of the input, e^a.

        """
        # Compute the exponential
        exp_value = float(operators.exp(a))

        # Save the result for the backward pass
        ctx.save_for_backward(exp_value)

        return exp_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: The gradient with respect to the input.

        """
        # Retrieve the exponential value saved during the forward pass
        exp_value = ctx.saved_values[0]

        # The derivative of e^x is e^x itself
        return d_output * exp_value


class LT(ScalarFunction):
    """Less-than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the less-than function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        """
        # Compute the less-than comparison
        result = 1.0 if a < b else 0.0

        # No need to save any values for backward pass
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for the less-than function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs (a, b).

        """
        # The derivative of the less-than function is always 0 for both inputs
        # because it's a step function (not continuous or differentiable)
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the equality function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        """
        # Compute the equality comparison
        result = 1.0 if a == b else 0.0

        # No need to save any values for backward pass
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for the equality function.

        Args:
        ----
            ctx (Context): The context containing saved values from the forward pass.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients with respect to the inputs (a, b).

        """
        # The derivative of the equality function is always 0 for both inputs
        # because it's a step function (not continuous or differentiable)
        return (0.0, 0.0)
