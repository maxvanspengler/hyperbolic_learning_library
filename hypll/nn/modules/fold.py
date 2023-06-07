from torch.nn import Module
from torch.nn.common_types import _size_2_t

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HUnfold(Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        manifold: Manifold,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> None:
        self.kernel_size = kernel_size
        self.manifold = manifold
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        check_if_man_dims_match(layer=self, man_dim=1, input=input)
        return self.manifold.unfold(
            input=input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
