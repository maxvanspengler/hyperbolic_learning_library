from functools import partial
from typing import Optional

from torch.nn import Module
from torch.nn.common_types import _size_2_t, _size_any_t
from torch.nn.functional import max_pool2d

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import (
    check_if_man_dims_match,
    check_if_manifolds_match,
    op_in_tangent_space,
)


class HAvgPool2d(Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        manifold: Manifold,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        use_midpoint: bool = False,
    ):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple) and len(kernel_size) == 2
            else (kernel_size, kernel_size)
        )
        self.manifold = manifold
        self.stride = stride if (stride is not None) else self.kernel_size
        self.padding = (
            padding if isinstance(padding, tuple) and len(padding) == 2 else (padding, padding)
        )
        self.use_midpoint = use_midpoint

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        check_if_man_dims_match(layer=self, man_dim=1, input=input)

        batch_size, channels, height, width = input.size()
        out_height = int((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        out_width = int((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        unfolded_input = self.manifold.unfold(
            input=input,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        per_kernel_view = unfolded_input.tensor.view(
            batch_size,
            channels,
            self.kernel_size[0] * self.kernel_size[1],
            unfolded_input.size(-1),
        )

        x = ManifoldTensor(data=per_kernel_view, manifold=self.manifold, man_dim=1)

        if self.use_midpoint:
            aggregates = self.manifold.midpoint(x=x, batch_dim=2)

        else:
            aggregates = self.manifold.frechet_mean(x=x, batch_dim=2)

        return ManifoldTensor(
            data=aggregates.tensor.reshape(batch_size, channels, out_height, out_width),
            manifold=self.manifold,
            man_dim=1,
        )


class _HMaxPoolNd(Module):
    def __init__(
        self,
        kernel_size: _size_any_t,
        manifold: Manifold,
        stride: Optional[_size_any_t] = None,
        padding: _size_any_t = 0,
        dilation: _size_any_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.manifold = manifold
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode


class HMaxPool2d(_HMaxPoolNd):
    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)

        # TODO: check if defining this partial func each forward pass is slow.
        # If it is, put this stuff inside the init or add kwargs to op_in_tangent_space.
        max_pool2d_partial = partial(
            max_pool2d,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )
        return op_in_tangent_space(op=max_pool2d_partial, manifold=self.manifold, input=input)
