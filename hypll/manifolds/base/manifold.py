from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.common_types import _size_2_t

# TODO: find a less hacky solution for this
if TYPE_CHECKING:
    from hypll.tensors import ManifoldParameter, ManifoldTensor, TangentTensor


class Manifold(Module, ABC):
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
    def cdist(self, x: ManifoldTensor, y: ManifoldTensor) -> Tensor:
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
    def frechet_mean(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def midpoint(
        self,
        x: ManifoldTensor,
        batch_dim: Union[int, list[int]] = 0,
        w: Optional[Tensor] = None,
        keepdim: bool = False,
    ) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def frechet_variance(
        self,
        x: ManifoldTensor,
        mu: Optional[ManifoldTensor] = None,
        batch_dim: Union[int, list[int]] = -1,
        keepdim: bool = False,
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

    @abstractmethod
    def flatten(self, x: ManifoldTensor, start_dim: int = 1, end_dim: int = -1) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def unfold(
        self,
        input: ManifoldTensor,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
    ) -> ManifoldTensor:
        raise NotImplementedError

    @abstractmethod
    def cat(
        self,
        manifold_tensors: Union[Tuple[ManifoldTensor, ...], List[ManifoldTensor]],
        dim: int = 0,
    ) -> ManifoldTensor:
        raise NotImplementedError
