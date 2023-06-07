from torch import broadcast_shapes, equal

from hypll.tensors import ManifoldTensor, TangentTensor


def check_dims_with_broadcasting(*args: ManifoldTensor) -> int:
    # Check if shapes can be broadcasted together
    shapes = [a.size() for a in args]
    try:
        broadcasted_shape = broadcast_shapes(*shapes)
    except RuntimeError:
        raise ValueError(f"Shapes of inputs were {shapes} and cannot be broadcasted together")

    # Find the manifold dimensions after broadcasting
    max_dim = len(broadcasted_shape)
    man_dims = []
    for a in args:
        if isinstance(a, ManifoldTensor):
            man_dims.append(a.man_dim + (max_dim - a.dim()))
        elif isinstance(a, TangentTensor):
            man_dims.append(a.broadcasted_man_dim + (max_dim - a.dim()))

    for md in man_dims[1:]:
        if man_dims[0] != md:
            raise ValueError("Manifold dimensions of inputs after broadcasting do not match.")

    return man_dims[0]


def check_tangent_tensor_positions(*args: TangentTensor) -> None:
    manifold_points = [a.manifold_points for a in args]
    if any([mp is None for mp in manifold_points]):
        if not all(mp is None for mp in manifold_points):
            raise ValueError(f"Some but not all tangent tensors are located at the origin")

    else:
        broadcasted_shape = broadcast_shapes(*[a.size() for a in args])
        broadcasted_manifold_points = [
            mp.tensor.broadcast_to(broadcasted_shape) for mp in manifold_points
        ]
        for bmp in broadcasted_manifold_points[1:]:
            if not equal(broadcasted_manifold_points[0], bmp):
                raise ValueError(
                    f"Tangent tensors are positioned at the different points on the manifold"
                )
