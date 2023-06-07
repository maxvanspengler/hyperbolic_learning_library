from collections.abc import Iterable
from typing import Union

from torch import max, no_grad, zeros_like
from torch.optim import Optimizer

from hypll.manifolds import Manifold
from hypll.manifolds.euclidean import Euclidean
from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor


class RiemannianAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[Union[ManifoldParameter, ManifoldTensor]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(RiemannianAdam, self).__init__(params=params, defaults=defaults)

    def step(self) -> None:
        # TODO: refactor this and add some comments, because it's currently unreadable
        with no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                lr = group["lr"]
                amsgrad = group["amsgrad"]
                for param in group["params"]:
                    if isinstance(param, ManifoldParameter):
                        manifold: Manifold = param.manifold
                        grad = param.grad
                        if grad is None:
                            continue
                        grad = ManifoldTensor(
                            data=grad, manifold=Euclidean(), man_dim=param.man_dim
                        )
                        state = self.state[param]

                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = zeros_like(param.tensor)
                            state["exp_avg_sq"] = zeros_like(param.tensor)
                            if amsgrad:
                                state["max_exp_avg_sq"] = zeros_like(param.tensor)

                        state["step"] += 1
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]

                        # TODO: check if this next line makes sense, because I don't think so
                        grad.tensor.add_(param.tensor, alpha=weight_decay)
                        grad = manifold.euc_to_tangent(x=param, u=grad)
                        exp_avg.mul_(betas[0]).add_(grad.tensor, alpha=1 - betas[0])
                        exp_avg_sq.mul_(betas[1]).add_(
                            manifold.inner(u=grad, v=grad, keepdim=True, safe_mode=False),
                            alpha=1 - betas[1],
                        )
                        bias_correction1 = 1 - betas[0] ** state["step"]
                        bias_correction2 = 1 - betas[1] ** state["step"]

                        if amsgrad:
                            max_exp_avg_sq = state["max_exp_avg_sq"]
                            max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                            denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                        else:
                            denom = exp_avg_sq.div(bias_correction2).sqrt_()

                        direction = -lr * exp_avg.div(bias_correction1) / denom.add_(eps)
                        direction = TangentTensor(
                            data=direction,
                            manifold_points=param,
                            manifold=manifold,
                            man_dim=param.man_dim,
                        )
                        exp_avg_man = TangentTensor(
                            data=exp_avg,
                            manifold_points=param,
                            manifold=manifold,
                            man_dim=param.man_dim,
                        )

                        new_param = manifold.expmap(direction)
                        exp_avg_new = manifold.transp(v=exp_avg_man, y=new_param)

                        param.tensor.copy_(new_param.tensor)
                        exp_avg.copy_(exp_avg_new.tensor)

                    else:
                        grad = param.grad
                        if grad is None:
                            continue

                        state = self.state[param]

                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = zeros_like(param)
                            state["exp_avg_sq"] = zeros_like(param)
                            if amsgrad:
                                state["max_exp_avg_sq"] = zeros_like(param)
                        state["step"] += 1
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        grad.add_(param, alpha=weight_decay)
                        exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                        exp_avg_sq.mul_(betas[1]).add_(
                            grad.square().sum(dim=-1, keepdim=True), alpha=1 - betas[1]
                        )
                        bias_correction1 = 1 - betas[0] ** state["step"]
                        bias_correction2 = 1 - betas[1] ** state["step"]
                        if amsgrad:
                            max_exp_avg_sq = state["max_exp_avg_sq"]
                            max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                            denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                        else:
                            denom = exp_avg_sq.div(bias_correction2).sqrt_()

                        direction = exp_avg.div(bias_correction1) / denom.add_(eps)

                        new_param = param - lr * direction
                        exp_avg_new = exp_avg

                        param.copy_(new_param)
                        exp_avg.copy_(exp_avg_new)
