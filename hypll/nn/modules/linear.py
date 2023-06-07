from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HLinear(Module):
    """Poincare fully connected linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        bias: bool = True,
    ) -> None:
        super(HLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.has_bias = bias

        # TODO: torch stores weights transposed supposedly due to efficiency
        # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/7
        # We may want to do the same
        self.z, self.bias = self.manifold.construct_dl_parameters(
            in_features=in_features, out_features=out_features, bias=self.has_bias
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.manifold.reset_parameters(self.z, self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)
        return self.manifold.fully_connected(x=x, z=self.z, bias=self.bias)
