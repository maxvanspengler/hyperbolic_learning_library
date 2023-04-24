from torch import empty, eye
from torch.nn import Module, Parameter
from torch.nn.init import normal_, zeros_

from hypdl.manifolds import Manifold
from hypdl.tensors import ManifoldTensor
from hypdl.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HLinear(Module):
    """Poincare fully connected linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        bias: bool = True,
        id_init: bool = True,
    ) -> None:
        super(HLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.has_bias = bias
        self.id_init = id_init

        self.z = Parameter(empty(in_features, out_features))
        if self.has_bias:
            self.bias = Parameter(empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO: this stuff depends on the manifold, so may need to move logic into there.
        if self.id_init:
            self.z = Parameter(1 / 2 * eye(self.in_features, self.out_features))
        else:
            normal_(
                self.z,
                mean=0,
                std=(2 * self.in_features * self.out_features) ** -0.5,
            )
        if self.has_bias:
            zeros_(self.bias)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)
        return self.manifold.fully_connected(x=x, z=self.z, bias=self.bias)
