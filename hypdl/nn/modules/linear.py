from torch import Tensor, empty, eye
from torch.nn import Module, Parameter
from torch.nn.init import normal_, zeros_

from ...manifolds import Manifold


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

    def forward(self, x: Tensor) -> Tensor:
        return self.manifold.fully_connected(x=x, z=self.z, bias=self.bias)
