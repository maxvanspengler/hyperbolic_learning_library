from collections.abc import Iterable
from typing import Union

from torch import no_grad
from torch.optim import Optimizer

from ..manifolds import Manifold
from ..tensors import ManifoldParameter, ManifoldTensor


class RiemannianSGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Union[ManifoldParameter, ManifoldTensor]],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        super(RiemannianSGD, self).__init__(params=params, defaults=defaults)

    def step(self) -> None:
        with no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                weight_decay = group["weight_decay"]
                nestorov = group["nestorov"]

                for param in group:
                    grad = param.grad
                    if grad is None:
                        continue
                    state = self.state[param]

                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()
                    manifold: Manifold = param.manifold

                    grad.add_(param, alpha=weight_decay)  # TODO: check if this makes sense
                    grad = manifold.euc_to_tangent(x=param, u=grad, dim=param.man_dim)
                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nestorov:
                            grad = grad.add_(momentum_buffer, alpha=momentum)
                        else:
                            grad = momentum_buffer

                        new_param = manifold.expmap(x=param, v=-lr * grad)
                        new_momentum_buffer = manifold.transp(
                            x=param, y=new_param, v=momentum_buffer
                        )
                        momentum_buffer.copy_(new_momentum_buffer)
                        param.copy_(new_param)
                    else:
                        new_param = manifold.expmap(x=param, v=-lr * grad)
                        param.copy_(new_param)
