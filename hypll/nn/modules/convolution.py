from torch.nn import Module
from torch.nn.common_types import _size_1_t, _size_2_t

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HConvolution2d(Module):
    """Applies a 2D convolution over a hyperbolic input signal.

    Attributes:
        in_channels:
            Number of channels in the input image.
        out_channels:
            Number of channels produced by the convolution.
        kernel_size:
            Size of the convolving kernel.
        manifold:
            Hyperbolic manifold of the tensors.
        bias:
            If True, adds a learnable bias to the output. Default: True
        stride:
            Stride of the convolution. Default: 1
        padding:
            Padding added to all four sides of the input. Default: 0
        id_init:
            Use identity initialization (True) if appropriate or use HNN++ initialization (False).

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        manifold: Manifold,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
    ) -> None:
        super(HConvolution2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple) and len(kernel_size) == 2
            else (kernel_size, kernel_size)
        )
        self.kernel_vol = self.kernel_size[0] * self.kernel_size[1]
        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.id_init = id_init
        self.has_bias = bias

        self.weights, self.bias = self.manifold.construct_dl_parameters(
            in_features=self.kernel_vol * in_channels,
            out_features=out_channels,
            bias=self.has_bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets parameter weights based on the manifold."""
        self.manifold.reset_parameters(weight=self.weights, bias=self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """Does a forward pass of the 2D convolutional layer.

        Args:
            x:
                Manifold tensor of shape (B, C_in, H, W) with manifold dimension 1.

        Returns:
            Manifold tensor of shape (B, C_in, H_out, W_out) with manifold dimension 1.

        Raises:
            ValueError: If the manifolds or manifold dimensions don't match.

        """
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)

        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        out_height = _output_side_length(
            input_side_length=height,
            kernel_size=self.kernel_size[0],
            padding=self.padding,
            stride=self.stride,
        )
        out_width = _output_side_length(
            input_side_length=width,
            kernel_size=self.kernel_size[1],
            padding=self.padding,
            stride=self.stride,
        )

        x = self.manifold.unfold(
            input=x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        x = self.manifold.fully_connected(x=x, z=self.weights, bias=self.bias)
        x = ManifoldTensor(
            data=x.tensor.reshape(batch_size, self.out_channels, out_height, out_width),
            manifold=x.manifold,
            man_dim=1,
        )
        return x


def _output_side_length(
    input_side_length: int, kernel_size: _size_1_t, padding: int, stride: int
) -> int:
    """Calculates the output side length of the kernel.

    Based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

    """
    if kernel_size > input_side_length:
        raise RuntimeError(
            f"Encountered invalid kernel size {kernel_size} "
            f"larger than input side length {input_side_length}"
        )
    if stride > input_side_length:
        raise RuntimeError(
            f"Encountered invalid stride {stride} "
            f"larger than input side length {input_side_length}"
        )
    return 1 + (input_side_length + 2 * padding - (kernel_size - 1) - 1) // stride
