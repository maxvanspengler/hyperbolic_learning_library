from pytest_mock import MockerFixture

from hll.manifolds.poincare_ball import Curvature


def test_curvature(mocker: MockerFixture) -> None:
    mocked_constraining_strategy = mocker.MagicMock()
    mocked_constraining_strategy.return_value = 1.33  # dummy value
    curvature = Curvature(value=1.0, constraining_strategy=mocked_constraining_strategy)
    assert curvature() == 1.33
    mocked_constraining_strategy.assert_called_once_with(1.0)
