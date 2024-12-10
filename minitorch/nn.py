from typing import Tuple

from . import operators
from .tensor import Tensor
from .tensor_functions import rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw

    # Make input contiguous at the start
    input = input.contiguous()

    return (
        input.view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D
    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns
    -------
        Pooled tensor

    """
    # Use tile to reshape the input tensor
    tiled_input, new_height, new_width = tile(input, kernel)

    # Calculate mean over the last dimension and ensure the output is properly shaped
    return (
        tiled_input.mean(dim=4)
        .contiguous()
        .view(tiled_input.shape[0], tiled_input.shape[1], new_height, new_width)
    )


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along dimension dim."""
    # Handle negative dimensions
    if dim < 0:
        dim = len(input.shape) + dim

    # Create a view where the target dim is last
    dims = list(range(input.dims))
    dims.append(dims.pop(dim))
    view = input.permute(*dims)

    # Get shape info
    shape = view.shape
    last_dim = shape[-1]
    other_dims = int(operators.prod(shape[:-1]))

    # Reshape to combine all other dimensions
    view = view.contiguous().view(other_dims, last_dim)

    # Apply max reduction
    out = view.zeros((other_dims,))
    for i in range(other_dims):
        out[i] = view[i, 0]  # Initialize with first element
        for j in range(1, last_dim):
            if view[i, j] > out[i]:
                out[i] = view[i, j]

    # Reshape back to original dimensions with reduced dim as size 1
    new_shape = list(input.shape)  # Use input.shape instead of shape
    new_shape[dim] = 1  # Set the reduced dimension to 1
    return out.view(*new_shape)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along dimension."""
    if dim < 0:
        dim = len(input.shape) + dim

    # Get max for numerical stability (already has correct shape for broadcasting)
    m = max(input, dim)

    # Compute exp(x - max(x)) for numerical stability
    exp_x = (input - m).exp()

    # Sum along the specified dimension (will keep dim)
    sum_exp = exp_x.sum(dim=dim)

    # Compute softmax
    out = exp_x / sum_exp
    return out.contiguous()


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log of softmax along dimension."""
    if dim < 0:
        dim = len(input.shape) + dim

    # Get max for numerical stability
    m = max(input, dim)

    # Compute exp(x - max(x)) for numerical stability
    exp_x = (input - m).exp()

    # Sum along the specified dimension (keep dim)
    sum_exp = exp_x.sum(dim=dim)

    # Compute log softmax directly
    return (input - m) - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D."""
    batch, channel, height, width = input.shape
    kh, kw = kernel

    # Calculate output dimensions
    new_height = height // kh
    new_width = width // kw

    # Create output tensor
    output = input.zeros((batch, channel, new_height, new_width))

    # Perform max pooling
    for b in range(batch):
        for c in range(channel):
            for h in range(new_height):
                for w in range(new_width):
                    # Get the current window
                    curr_max = input[b, c, h * kh, w * kw]
                    for i in range(kh):
                        for j in range(kw):
                            val = input[b, c, h * kh + i, w * kw + j]
                            curr_max = operators.max(curr_max, val)
                    output[b, c, h, w] = curr_max

    return output


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise."""
    if ignore:
        return input

    if not (0.0 <= rate <= 1.0):
        raise ValueError("Dropout rate must be in [0, 1]")

    if rate == 1.0:
        return input.zeros(input.shape)

    # Generate random mask
    mask = rand(input.shape, backend=input.backend) > rate

    # Scale output
    scale = 1.0 / (1.0 - rate)
    return mask * input * scale
