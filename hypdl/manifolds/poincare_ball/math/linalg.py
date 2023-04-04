from typing import Optional

import torch


def poincare_mlr(
    x: torch.Tensor,
    z: torch.Tensor,
    r: Optional(torch.Tensor),
    c: torch.Tensor,
) -> torch.Tensor:
    """
    The Poincare multinomial logistic regression (MLR) operation.

    Parameters
    ----------
    x : tensor
        contains input values
    z : tensor
        contains the hyperbolic vectors describing the hyperplane orientations
    r : tensor
        contains the hyperplane offsets
    c : tensor
        curvature of the Poincare disk

    Returns
    -------
    tensor
        signed distances of input w.r.t. the hyperplanes, denoted by v_k(x) in
        the HNN++ paper
    """
    c_sqrt = c.sqrt()
    lam = 2 * (1 - c * x.pow(2).sum(dim=-1, keepdim=True))
    z_norm = z.norm(dim=0).clamp_min(1e-15)

    # Computation can be simplified if there is no offset
    if r is None:
        return 2 * z_norm / c_sqrt * torch.asinh(c_sqrt * lam / z_norm * torch.matmul(x, z))
    else:
        two_csqrt_r = 2.0 * c_sqrt * r
        return (
            2
            * z_norm
            / c_sqrt
            * torch.asinh(
                c_sqrt * lam / z_norm * torch.matmul(x, z) * two_csqrt_r.cosh()
                - (lam - 1) * two_csqrt_r.sinh()
            )
        )


def poincare_fully_connected(
    x: torch.Tensor,
    z: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    The Poincare fully connected layer operation.

    Parameters
    ----------
    x : tensor
        contains the layer inputs
    z : tensor
        contains the hyperbolic vectors describing the hyperplane orientations
    bias : tensor
        contains the biases (hyperplane offsets)
    c : tensor
        curvature of the Poincare disk

    Returns
    -------
    tensor
        Poincare FC transformed hyperbolic tensor, commonly denoted by y
    """
    c_sqrt = c.sqrt()
    x = poincare_mlr(x=x, z=z, r=bias, c=c)
    x = (c_sqrt * x).sinh() / c_sqrt
    return x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt())
