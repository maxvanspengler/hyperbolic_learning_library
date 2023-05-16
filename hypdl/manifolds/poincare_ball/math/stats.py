from typing import Optional, Union

import torch
import torch.nn as nn

from .diffgeom import dist

_TOLEPS = {torch.float32: 1e-6, torch.float64: 1e-12}


class FrechetMean(torch.autograd.Function):
    """
    This implementation is copied mostly from:
        https://github.com/CUVL/Differentiable-Frechet-Mean.git

    which is itself based on the paper:
        https://arxiv.org/abs/2003.00335

    Both by Aaron Lou (et al.)
    """

    @staticmethod
    def forward(ctx, x, w, K):
        mean = frechet_ball_forward(x, w, K, rtol=_TOLEPS[x.dtype], atol=_TOLEPS[x.dtype])
        ctx.save_for_backward(x, mean, w, K)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K = ctx.saved_tensors
        dx, dw, dK = frechet_ball_backward(X, mean, grad_output, w, K)
        return dx, dw, dK, None


def frechet_mean(
    x: torch.Tensor,
    c: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO: FrechetMean.apply should take c instead of -c: need to rewrite the computation a bit.
    if w is None:
        w = torch.ones(x.shape[:-1]).to(x)
    return FrechetMean.apply(x, w, -c)


def midpoint(
    x: torch.Tensor,
    c: torch.Tensor,
    man_dim: int = -1,
    batch_dim: Union[int, list[int]] = 0,
    w: Optional[torch.Tensor] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    lambda_x = 2 / (1 - c * x.pow(2).sum(dim=man_dim, keepdim=True)).clamp_min(1e-15)
    if w is None:
        numerator = (lambda_x * x).sum(dim=batch_dim, keepdim=True)
        denominator = (lambda_x - 1).sum(dim=batch_dim, keepdim=True)
    else:
        # TODO: test weights
        numerator = (lambda_x * w * x).sum(dim=batch_dim, keepdim=True)
        denominator = ((lambda_x - 1) * w).sum(dim=batch_dim, keepdim=True)

    frac = numerator / denominator.clamp_min(1e-15)
    midpoint = 1 / (1 + (1 - c * frac.pow(2).sum(dim=man_dim, keepdim=True)).sqrt()) * frac
    if not keepdim:
        midpoint = midpoint.squeeze(dim=batch_dim)

    return midpoint


def frechet_variance(
    x: torch.Tensor,
    mu: torch.Tensor,
    c: torch.Tensor,
    dim: int = -1,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args
    ----
        x (tensor): points of shape [..., points, dim]
        mu (tensor): mean of shape [..., dim]
        w (tensor): weights of shape [..., points]

        where the ... of the three variables match

    Returns
    -------
        tensor of shape [...]
    """
    distance = dist(x=x, y=mu, k=c, dim=dim)
    distance = distance.pow(2)

    if w is None:
        return distance.mean(dim=-1)
    else:
        return (distance * w).sum(dim=-1)


def l_prime(y: torch.Tensor) -> torch.Tensor:
    cond = y < 1e-12
    val = 4 * torch.ones_like(y)
    ret = torch.where(cond, val, 2 * (1 + 2 * y).acosh() / (y.pow(2) + y).sqrt())
    return ret


def frechet_ball_forward(
    X: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor = torch.Tensor([-1]),
    max_iter: int = 1000,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> torch.Tensor:
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        w (tensor): weights of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        frechet mean (tensor): shape [..., dim]
    """
    mu = X[..., 0, :].clone()

    x_ss = X.pow(2).sum(dim=-1)

    mu_prev = mu
    iters = 0
    for _ in range(max_iter):
        mu_ss = mu.pow(2).sum(dim=-1)
        xmu_ss = (X - mu.unsqueeze(-2)).pow(2).sum(dim=-1)

        alphas = l_prime(-K * xmu_ss / ((1 + K * x_ss) * (1 + K * mu_ss.unsqueeze(-1)))) / (
            1 + K * x_ss
        )

        alphas = alphas * w

        c = (alphas * x_ss).sum(dim=-1)
        b = (alphas.unsqueeze(-1) * X).sum(dim=-2)
        a = alphas.sum(dim=-1)

        b_ss = b.pow(2).sum(dim=-1)

        eta = (a - K * c - ((a - K * c).pow(2) + 4 * K * b_ss).sqrt()) / (2 * (-K) * b_ss)

        mu = eta.unsqueeze(-1) * b

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)
        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    return mu


def darcosh(x):
    cond = x < 1 + 1e-7
    x = torch.where(cond, 2 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 * (x).acosh() / torch.sqrt(x**2 - 1), x)
    return x


def d2arcosh(x):
    cond = x < 1 + 1e-7
    x = torch.where(cond, -2 / 3 * torch.ones_like(x), x)
    x = torch.where(
        ~cond,
        2 / (x**2 - 1) - 2 * x * x.acosh() / ((x**2 - 1) ** (3 / 2)),
        x,
    )
    return x


def grad_var(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        grad (tensor): gradient of variance [..., dim]
    """
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)

    first_term = (X - yl) / Dl
    sec_term = K / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    return -(4 * darcosh(vl) * w.unsqueeze(-1) * (first_term + sec_term)).sum(dim=-2)


def inverse_hessian(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        inv_hess (tensor): inverse hessian of [..., points, dim, dim]
    """
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)
    vll = vl.unsqueeze(-1)

    """
    \partial T/ \partial y
    """
    first_const = -8 * (K**2) * xnorm / D.pow(2)
    matrix_val = (first_const.unsqueeze(-1) * yl).unsqueeze(-1) * (X - yl).unsqueeze(-2)
    first_term = matrix_val + matrix_val.transpose(-1, -2)

    sec_const = -16 * (K**3) * xnorm.pow(2) / D.pow(3) * xynorm
    sec_term = (sec_const.unsqueeze(-1) * yl).unsqueeze(-1) * yl.unsqueeze(-2)

    third_const = -4 * K / D + 4 * (K**2) * xnorm / D.pow(2) * xynorm
    third_term = third_const.reshape(*third_const.shape, 1, 1) * torch.eye(y.shape[-1]).to(
        X
    ).reshape((1,) * len(third_const.shape) + (y.shape[-1], y.shape[-1]))

    Ty = first_term + sec_term + third_term

    """
    T
    """

    first_term = K / Dl * (X - yl)
    sec_term = K.pow(2) / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    T = 4 * (first_term + sec_term)

    """
    inverse of shape [..., points, dim, dim]
    """
    first_term = d2arcosh(vll) * T.unsqueeze(-1) * T.unsqueeze(-2)
    sec_term = darcosh(vll) * Ty
    hessian = ((first_term + sec_term) * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=-3) / -K
    inv_hess = torch.inverse(hessian)
    return inv_hess


def frechet_ball_backward(
    X: torch.Tensor,
    y: torch.Tensor,
    grad: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor]:
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        grad (tensor): gradient
        K (float): curvature (must be negative)

    Returns
    -------
        gradients (tensor, tensor, tensor):
            gradient of X [..., points, dim], weights [..., dim], curvature []
    """
    if not torch.is_tensor(K):
        K = torch.tensor(K).to(X)

    with torch.no_grad():
        inv_hess = inverse_hessian(X, y, w=w, K=K)

    with torch.enable_grad():
        # clone variables
        X = nn.Parameter(X.detach())
        y = y.detach()
        w = nn.Parameter(w.detach())
        K = nn.Parameter(K)

        grad = (inv_hess @ grad.unsqueeze(-1)).squeeze()
        gradf = grad_var(X, y, w, K)
        dx, dw, dK = torch.autograd.grad(-gradf.squeeze(), (X, w, K), grad)

    return dx, dw, dK
