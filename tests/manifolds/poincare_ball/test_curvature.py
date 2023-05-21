from pytest_mock import MockerFixture
from torch.nn.functional import softplus

from hypdl.manifolds.poincare_ball.curvature import Curvature


def test_curvature(mocker: MockerFixture) -> None:
    mocked_non_negative_function = mocker.MagicMock()
    mocked_non_negative_function.return_value = 1.33  # dummy value
    curvature = Curvature(_c=1.0, learnable=False, non_negative_function=non_negative_function)
    c = curvature()
    assert c == 1.33
    mocked_non_negative_function.assert_called_once_with(1.0)
