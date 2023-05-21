from typing import Callable

import torch
from torch import Tensor, as_tensor
from torch.nn import Module
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class Curvature(Module):
    def __init__(
        self,
        _c: float = 1.0,
        learnable: bool = True,
        positive_function: Callable[[Tensor], Tensor] = softplus,
    ):
        super(Curvature, self).__init__()
        self._c = Parameter(as_tensor(_c, dtype=torch.float32), requires_grad=learnable)
        self._positive_function = positive_function

    def forward(self) -> Tensor:
        return self._positive_function(self._c)
