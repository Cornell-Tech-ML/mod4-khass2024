"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the negation of a tensor element-wise.

        Args:
        ----
            ctx: Context object to save info for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with negated values

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for negation operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes multiplicative inverse (1/x) element-wise.

        Args:
        ----
            ctx: Context object to save info for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with inverse values

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for inverse operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Adds two tensors element-wise.

        Args:
        ----
            ctx: Context object to save info for backward pass
            t1: First input tensor
            t2: Second input tensor

        Returns:
        -------
            Sum of the input tensors

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradient for addition operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of gradients with respect to inputs

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Multiplies two tensors element-wise.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: First input tensor
            b: Second input tensor

        Returns:
        -------
            Element-wise product of the input tensors

        """
        # ASSIGN2.3
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradients for multiplication operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of gradients with respect to inputs

        """
        # ASSIGN2.4
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )
        # END ASSIGN2.4


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies sigmoid function element-wise.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with sigmoid applied element-wise

        """
        # ASSIGN2.3
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for sigmoid operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        # ASSIGN2.4
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output
        # END ASSIGN2.4


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies ReLU function element-wise.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with ReLU applied element-wise

        """
        # ASSIGN2.3
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for ReLU operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        # ASSIGN2.4
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)
        # END ASSIGN2.4


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies natural logarithm element-wise.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with natural log applied element-wise

        """
        # ASSIGN2.3
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for logarithm operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        # ASSIGN2.4
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)
        # END ASSIGN2.4


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Applies exponential function element-wise.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            t1: Input tensor

        Returns:
        -------
            Tensor with exponential applied element-wise

        """
        # ASSIGN2.3
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes gradient for exponential operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Gradient with respect to input

        """
        # ASSIGN2.4
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)
        # END ASSIGN2.4


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes element-wise less than comparison.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: First input tensor
            b: Second input tensor

        Returns:
        -------
            Boolean tensor with element-wise less than results

        """
        # ASSIGN2.3
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradient for less than operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of zero gradients for both inputs

        """
        # ASSIGN2.4
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)
        # END ASSIGN2.4


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes element-wise equality comparison.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: First input tensor
            b: Second input tensor

        Returns:
        -------
            Boolean tensor with element-wise equality results

        """
        # ASSIGN2.3
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes gradient for equality operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of zero gradients for both inputs

        """
        # ASSIGN2.4
        a_shape, b_shape = ctx.saved_values
        return zeros(a_shape), zeros(b_shape)
        # END ASSIGN2.4


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Checks if elements are approximately equal.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: First input tensor
            b: Second input tensor

        Returns:
        -------
            Boolean tensor indicating which elements are close

        """
        # ASSIGN2.3
        return a.f.is_close_zip(a, b)
        # END ASSIGN2.3


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permutes tensor dimensions.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: Input tensor
            order: Tensor specifying new dimension order

        Returns:
        -------
            Tensor with permuted dimensions

        """
        # ASSIGN2.3
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))
        # END ASSIGN2.3

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes gradient for permute operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of gradient for input tensor and zero for order tensor

        """
        # ASSIGN2.4
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Sums tensor elements along specified dimension.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: Input tensor
            dim: Dimension to sum over

        Returns:
        -------
            Tensor with sum along specified dimension

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes gradient for sum operation.

        Args:
        ----
            ctx: Context object with saved tensors
            grad_output: Upstream gradient

        Returns:
        -------
            Tuple of gradient for input tensor and zero for dim tensor

        """
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshapes tensor to specified shape.

        Args:
        ----
            ctx: Context object to save tensors for backward pass
            a: Input tensor
            shape: New shape

        Returns:
        -------
            Reshaped tensor

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes gradient using central difference approximation.

    Args:
    ----
        f: Function to compute gradient for
        *vals: Input tensors
        arg: Which argument to compute gradient with respect to
        epsilon: Small value for finite difference
        ind: Index to compute gradient at

    Returns:
    -------
        Approximate gradient value at specified index

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
