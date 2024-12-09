from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, Set, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    # Create a list of values to pass to the function
    vals_list = list(vals)

    # Calculate f(x + epsilon)
    vals_list[arg] += epsilon
    f_plus = f(*vals_list)

    # Calculate f(x - epsilon)
    vals_list[arg] -= 2 * epsilon  # Subtract 2*epsilon to get to (x - epsilon)
    f_minus = f(*vals_list)

    # Compute the central difference
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol defining the interface for a variable in the computation graph."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable.

        Args:
        ----
            x (Any): The derivative to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable.

        Returns
        -------
            int: A unique identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node in the computation graph.

        Returns
        -------
            bool: True if the variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is a constant.

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph.

        Returns
        -------
            Iterable["Variable"]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples containing parent variables and their gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """

    def dfs(node: Variable, visited: Set[int], result: List[Variable]) -> None:
        """Depth-first search to perform topological sort."""
        if node.unique_id in visited or node.is_constant():
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            dfs(parent, visited, result)
        result.append(node)

    visited: Set[int] = set()
    result: List[Variable] = []
    dfs(variable, visited, result)
    return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The variable to start backpropagation from.
        deriv: The derivative of the output with respect to the variable.

    """
    sorted_variables = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in sorted_variables:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            for parent, grad in var.chain_rule(derivatives[var.unique_id]):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = grad
                else:
                    derivatives[parent.unique_id] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors.

        Returns
        -------
            Tuple[Any, ...]: A tuple of saved tensor values.

        """
        return self.saved_values
