import pytest

from hypll.nn.modules import convolution


def test__output_side_length() -> None:
    assert (
        convolution._output_side_length(input_side_length=32, kernel_size=3, stride=1, padding=0)
        == 30
    )
    assert (
        convolution._output_side_length(input_side_length=32, kernel_size=3, stride=2, padding=0)
        == 15
    )


def test__output_side_length__padding() -> None:
    assert (
        convolution._output_side_length(input_side_length=32, kernel_size=3, stride=1, padding=1)
        == 32
    )
    assert (
        convolution._output_side_length(input_side_length=32, kernel_size=3, stride=2, padding=1)
        == 16
    )
    # Reproduce https://github.com/maxvanspengler/hyperbolic_learning_library/issues/33
    assert (
        convolution._output_side_length(input_side_length=224, kernel_size=11, stride=4, padding=2)
        == 55
    )


def test__output_side_length__raises() -> None:
    with pytest.raises(RuntimeError) as e:
        convolution._output_side_length(input_side_length=32, kernel_size=33, stride=1, padding=0)
    with pytest.raises(RuntimeError) as e:
        convolution._output_side_length(input_side_length=32, kernel_size=3, stride=33, padding=0)
