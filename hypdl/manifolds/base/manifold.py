from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module


class Manifold(Module, ABC):
    # TODO: Fix return types of methods for all manifolds
    def __init__(self) -> None:
        super(Manifold, self).__init__()

    @property
    @abstractmethod
    def c(self):
        raise NotImplementedError

    @abstractmethod
    def mobius_add(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def project(self, x: Tensor, dim: int = -1, eps: float = -1.0) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def expmap0(self, v: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def logmap0(self, y: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def expmap(self, x: Tensor, v: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def logmap(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def gyration(self, u: Tensor, v: Tensor, w: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def transp(self, x: Tensor, y: Tensor, v: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def dist(self, x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def euc_to_tangent(x: Tensor, u: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def mlr(self, x: Tensor, z: Tensor, r: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def fully_connected(self, x: Tensor, z: Tensor, bias: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def frechet_mean(self, x: Tensor, w: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def frechet_variance(
        self, x: Tensor, mu: Tensor, dim: int = -1, w: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError
