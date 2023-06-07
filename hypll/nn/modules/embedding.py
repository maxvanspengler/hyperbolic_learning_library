from torch import Tensor, empty, no_grad, normal, ones_like, zeros_like
from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor


class HEmbedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold: Manifold,
    ) -> None:
        super(HEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold

        self.weight = ManifoldParameter(
            data=empty((num_embeddings, embedding_dim)), manifold=manifold, man_dim=-1
        )
        self.reset_parameters()

    def reset_parameters(self):
        with no_grad():
            new_weight = normal(
                mean=zeros_like(self.weight.tensor),
                std=ones_like(self.weight.tensor),
            )
            new_weight = TangentTensor(
                data=new_weight,
                manifold_points=None,
                manifold=self.manifold,
                man_dim=-1,
            )
            self.weight.copy_(self.manifold.expmap(new_weight).tensor)

    def forward(self, input: Tensor) -> ManifoldTensor:
        return self.weight[input]
