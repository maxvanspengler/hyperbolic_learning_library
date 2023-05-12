from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

from torch import Tensor
from torch.nn import Module, Parameter

# TODO: find a less hacky solution for this
if TYPE_CHECKING:
    from hypdl.tensors import ManifoldParameter, ManifoldTensor, TangentTensor


class Manifold(Module, ABC):
    # TODO: Fix return types of methods for all manifolds
    def __init__(self) -> None:
        super(Manifold, self).__init__()

    @abstractmethod
    def project(self, x: ManifoldTensor, eps: float = -1.0) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def expmap(self, v: TangentTensor) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def logmap(self, x: Optional[ManifoldTensor], y: ManifoldTensor) -> TangentTensor:
        raise NotImplementedError

    @abstractmethod
    def transp(self, v: TangentTensor, y: ManifoldTensor) -> TangentTensor:
        raise NotImplementedError

    @abstractmethod
    def dist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def euc_to_tangent(self, x: ManifoldTensor, u: ManifoldTensor) -> TangentTensor:
        raise NotImplementedError

    @abstractmethod
    def hyperplane_dists(self, x: ManifoldTensor, z: ManifoldTensor, r: Optional[Tensor]) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def fully_connected(
        self, x: ManifoldTensor, z: ManifoldTensor, bias: Optional[Tensor]
    ) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def frechet_mean(self, x: ManifoldTensor, w: Optional[Tensor] = None) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def frechet_variance(
        self, x: ManifoldTensor, mu: ManifoldTensor, w: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def construct_dl_parameters(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> Union[ManifoldParameter, tuple[ManifoldParameter, Parameter]]:
        # TODO: make an annotation object for the return type of this method
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self, weight: ManifoldParameter, bias: Parameter) -> None:
        raise NotImplementedError
