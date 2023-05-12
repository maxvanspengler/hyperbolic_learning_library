import unittest

import torch

from hypdl.manifolds import PoincareBall
from hypdl.tensors import ManifoldTensor, TangentTensor


class TestManifoldTensor(unittest.TestCase):
    def setUp(self):
        self.man_tensor = ManifoldTensor(
            data=[
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            manifold=PoincareBall(),
            man_dim=-1,
            requires_grad=True,
        )

        self.tan_tensor = TangentTensor(
            data=[
                [
                    [3.0, 2.0],
                    [1.0, 2.0],
                ],
                [
                    [1.0, 4.0],
                    [4.0, 1.0],
                ],
                [
                    [7.0, 2.0],
                    [3.0, 4.0],
                ],
            ],
            manifold_points=self.man_tensor,
            manifold=self.man_tensor.manifold,
            man_dim=-1,
        )

    def test_attributes(self):
        # Check if the standard attributes are set correctly
        # TODO: fix this once __eq__ has been implemented on manifolds
        self.assertTrue(isinstance(self.man_tensor.manifold, PoincareBall))
        self.assertEqual(self.man_tensor.man_dim, 1)

        # Check if non-callable attributes are taken from tensor attribute
        self.assertTrue(self.man_tensor.is_cpu)

    def test_device_methods(self):
        # Check if we can move the manifold tensor to the gpu while keeping it intact
        self.man_tensor = self.man_tensor.cuda()
        self.assertTrue(isinstance(self.man_tensor, ManifoldTensor))
        self.assertTrue(self.man_tensor.is_cuda)
        self.assertTrue(isinstance(self.man_tensor.manifold, PoincareBall))
        self.assertEqual(self.man_tensor.man_dim, 1)

        # And move it back to the cpu
        self.man_tensor = self.man_tensor.cpu()
        self.assertTrue(isinstance(self.man_tensor, ManifoldTensor))
        self.assertTrue(self.man_tensor.is_cpu)
        self.assertTrue(isinstance(self.man_tensor.manifold, PoincareBall))
        self.assertEqual(self.man_tensor.man_dim, 1)

    def test_torch_ops(self):
        # We want torch functons to raise an error
        with self.assertRaises(TypeError):
            torch.norm(self.man_tensor)

        # Same for torch.Tensor methods (callable attributes)
        with self.assertRaises(AttributeError):
            self.man_tensor.mean()


if __name__ == "__main__":
    unittest.main()
