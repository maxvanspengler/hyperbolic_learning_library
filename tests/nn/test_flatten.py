import pytest
from pytest_mock import MockerFixture

from hll.nn import HFlatten


def test_hflatten(mocker: MockerFixture) -> None:
    hflatten = HFlatten(start_dim=1, end_dim=-1)
    mocked_manifold_tensor = mocker.MagicMock()
    flattened = hflatten(mocked_manifold_tensor)
    mocked_manifold_tensor.flatten.assert_called_once_with(start_dim=1, end_dim=-1)
