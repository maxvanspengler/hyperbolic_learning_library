from typing import Callable

import torch
from torch import Tensor, as_tensor
from torch.nn import Module
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class Curvature(Module):
    """Class representing curvature of a manifold.

    Attributes:
        value:
            Learnable parameter indicating curvature of the manifold. The actual
            curvature is calculated as constraining_strategy(value).
        constraining_strategy:
            Function applied to the curvature value in order to constrain the
            curvature of the manifold. By default uses softplus to guarantee
            positive curvature.

    """

    def __init__(
        self,
        value: float = 1.0,
        constraining_strategy: Callable[[Tensor], Tensor] = softplus,
    ):
        super(Curvature, self).__init__()
        self.value = Parameter(as_tensor(value, dtype=torch.float32))
        self.constraining_strategy = constraining_strategy

    def forward(self) -> Tensor:
        """Returns curvature calculated as constraining_strategy(value)."""
        return self.constraining_strategy(self.value)
