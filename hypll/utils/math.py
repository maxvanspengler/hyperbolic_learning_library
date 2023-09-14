from math import exp, lgamma
from typing import Union

import torch


def beta_func(
    a: Union[float, torch.Tensor], b: Union[float, torch.Tensor]
) -> Union[float, torch.Tensor]:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return exp(lgamma(a) + lgamma(b) - lgamma(a + b))
