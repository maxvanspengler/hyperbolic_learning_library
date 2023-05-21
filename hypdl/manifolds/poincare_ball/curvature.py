from typing import Callable

from torch import Tensor, as_tensor
from torch.nn import Module
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


class Curvature(Module):
    def __init__(
        self,
        _c: float = 1.0,
        learnable: bool = True,
        non_negative_function: Callable[[Tensor], Tensor] = softplus,
    ):
        self._c = Parameter(as_tensor(_c, dtype=float32), requires_grad=learnable)
        self._non_negative_function = non_negative_function

    def forward(self) -> Tensor:
        return self._non_negative_function(self._c)
