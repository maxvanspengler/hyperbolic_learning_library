from torch.nn import Module
from torch.nn.common_types import _size_2_t

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor
from hypdl.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HConvolution2d(Module):
    """Hyperbolic 2 dimensional convolution layer"""

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
        self.kernel_size = kernel_size if len(kernel_size) == 2 else (kernel_size, kernel_size)
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
        self.manifold.reset_parameters(weight=self.weights, bias=self.bias)

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
        out_height = (height - self.kernel_size[0] + 1 + 2 * self.padding) // self.stride
        out_width = (width - self.kernel_size[1] + 1 + 2 * self.padding) // self.stride

        x = self.manifold.unfold(
            input=x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        x = self.manifold.fully_connected(x=x, z=self.weights, bias=self.bias)
        x = x.transpose(1, 2)
        x = ManifoldTensor(
            data=x.tensor.reshape(batch_size, self.out_channels, out_height, out_width),
            manifold=x.manifold,
            man_dim=1,
        )
        return x
