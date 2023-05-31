from functools import partial
from typing import Optional

from torch.nn import Module
from torch.nn.common_types import _size_2_t, _size_any_t
from torch.nn.functional import max_pool2d

from hll.manifolds import Manifold
from hll.tensors import ManifoldTensor
from hll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


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
