from torch.nn import Module, Sequential

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_manifolds_match


class TangentSequential(Module):
    def __init__(self, seq: Sequential, manifold: Manifold) -> None:
        super(TangentSequential, self).__init__()
        self.seq = seq
        self.manifold = manifold

    def forward(self, input: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=input)
        man_dim = input.man_dim

        input = self.manifold.logmap(x=None, y=input)
        for module in self.seq:
            input.tensor = module(input.tensor)
        return self.manifold.expmap(input)
