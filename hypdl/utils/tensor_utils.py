from hypdl.tensors.manifold_tensor import ManifoldTensor


def check_dims_with_broadcasting(*args: ManifoldTensor) -> int:
    max_dim = max(a.tensor.dim() for a in args)
    man_dims = [a.man_dim + (max_dim - a.tensor.dim()) for a in args]
    for md in man_dims[1:]:
        if man_dims[0] != md:
            raise ValueError("Manifold dimensions of inputs after broadcasting do not match.")

    return man_dims[0]
