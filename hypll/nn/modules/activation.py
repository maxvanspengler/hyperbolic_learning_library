from torch.nn import Module
from torch.nn.functional import relu

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match, op_in_tangent_space


class HReLU(Module):
    def __init__(self, manifold: Manifold) -> None:
        super(HReLU, self).__init__()
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        return op_in_tangent_space(
            op=relu,
            manifold=self.manifold,
            input=input,
        )
