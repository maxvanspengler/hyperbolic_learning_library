from typing import Tuple

from torch import empty, eye
from torch.nn import Module, Parameter, Unfold
from torch.nn.init import normal_, zeros_

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor
from hypdl.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match
from hypdl.utils.math import beta_func


class HConvolution2d(Module):
    """Hyperbolic 2 dimensional convolution layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dims: Tuple[int, int],
        manifold: Manifold,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
    ) -> None:
        super(HConvolution2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_dims = kernel_dims
        self.kernel_size = kernel_dims[0] * kernel_dims[1]
        self.manifold = manifold
        self.stride = stride
        self.padding = padding
        self.id_init = id_init

        self.unfold = Unfold(
            kernel_size=kernel_dims,
            padding=padding,
            stride=stride,
        )

        self.has_bias = bias
        if bias:
            self.bias = Parameter(empty(out_channels))
        self.weights = Parameter(empty(self.kernel_size * in_channels, out_channels))

        self.reset_parameters()

        # Create beta's for concatenating receptive field features
        # TODO: Move concatenation logic into manifold
        self.beta_ni = beta_func(self.in_channels / 2, 1 / 2)
        self.beta_n = beta_func(self.in_channels * self.kernel_size / 2, 1 / 2)

    def reset_parameters(self) -> None:
        # TODO: this stuff depends on the manifold, so may need to move logic into there.
        if self.id_init:
            self.weights = Parameter(
                1 / 2 * eye(self.kernel_size * self.in_channels, self.out_channels)
            )
        else:
            normal_(
                self.weights,
                mean=0,
                std=(2 * self.in_channels * self.kernel_size * self.out_channels) ** -0.5,
            )
        if self.has_bias:
            zeros_(self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """
        Forward pass of the 2 dimensional convolution layer

        Parameters
        ----------
        x : tensor (height, width, batchsize, input channels)
            contains the layer inputs

        Returns
        -------
        tensor (height, width, batchsize, output channels)
        """
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=1, input=x)

        batch_size, height, width = x.size(0), x.size(2), x.size(3)
        out_height = (height - self.kernel_dims[0] + 1 + 2 * self.padding) // self.stride
        out_width = (width - self.kernel_dims[1] + 1 + 2 * self.padding) // self.stride

        x = self.manifold.logmap0(x, dim=1)
        x = x * self.beta_n / self.beta_ni
        x = self.unfold(x)
        x = x.transpose(1, 2)
        x = self.manifold.expmap0(x, dim=-1)
        x = self.manifold.fully_connected(x=x, z=self.weights, bias=self.bias)
        x = x.transpose(1, 2).reshape(batch_size, self.out_channels, out_height, out_width)
        return x
