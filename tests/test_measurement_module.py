import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ncs.measurement_module import (
    create_gaussian_operator,
    create_measurement_operators,
    create_subsampling_operator,
)


def test_create_subsampling_operator():
    subsample_op, upsample_op = create_subsampling_operator(10, 5, 123)

    subsampled_signal = subsample_op(np.arange(10))
    assert_array_equal(subsampled_signal, [0, 4, 7, 8, 9])

    upsampled_signal = upsample_op([1, 2, 3, 4, 5])
    assert_array_equal(upsampled_signal, [1, 0, 0, 0, 2, 0, 0, 3, 4, 5])

    # Verifying ops are pseudoinverse
    up_subsampled_signal = subsample_op(upsample_op([1, 2, 3, 4, 5]))
    assert_array_equal(up_subsampled_signal, [1, 2, 3, 4, 5])

    # Verify adjoint property: <Ax, y> = <x, A^T y>
    # For any vectors x and y holds:
    # measurement_op(x) · y == x · adjoint_op(y)
    signal = np.arange(10)
    y = np.array([1, 2, 3, 4, 5])
    left_side = np.dot(subsample_op(signal), y)
    right_side = np.dot(signal, upsample_op(y))
    assert_array_equal(left_side, right_side)


def test_create_gaussian_operator():
    measure_op, adjoint_op = create_gaussian_operator(10, 5, 123)

    signal = np.arange(10)
    measurement = measure_op(signal)

    assert_allclose(measurement, [1.7, 3.8, 12.5, 7.5, 1.7], atol=0.1)

    upscaled = adjoint_op([1, 2, 3, 4, 5])
    assert_allclose(
        upscaled, [-2.3, -5.4, 9.1, -3.2, 0.1, -0.9, 4.2, 3.3, 7, -1.9], atol=0.1
    )

    # Verify adjoint property: <Ax, y> = <x, A^T y>
    # For any vectors x and y holds:
    # measurement_op(x) · y == x · adjoint_op(y)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    left_side = np.dot(measure_op(signal), y)
    right_side = np.dot(signal, adjoint_op(y))
    assert_allclose(left_side, right_side)


def test_create_measurement_operator_validation():
    with pytest.raises(ValueError, match="m must be less than n"):
        create_measurement_operators("subsampling", n=10, m=10, seed=123)

    with pytest.raises(ValueError, match="m must be less than n"):
        create_measurement_operators("gaussian", n=10, m=15)

    with pytest.raises(
        ValueError, match="Measurement mode invalid_mode is not supported"
    ):
        create_measurement_operators("invalid_mode", n=10, m=5)


def test_create_measurement_operator_passes_correct_operators():
    # Test subsampling operators
    subsample_op, upsample_op = create_measurement_operators(
        "subsampling", n=10, m=5, seed=123
    )
    signal = np.arange(10)
    subsampled = subsample_op(signal)
    assert_array_equal(subsampled, [0, 4, 7, 8, 9])

    upsampled = upsample_op(subsampled)
    assert upsampled.shape == (10,)

    # Test gaussian operators
    measure_op, adjoint_op = create_measurement_operators(
        "gaussian", n=10, m=5, seed=123
    )
    measurements = measure_op(signal)
    assert measurements.shape == (5,)
    assert_allclose(measurements, [1.7, 3.8, 12.5, 7.5, 1.7], atol=0.1)

    reconstructed = adjoint_op(measurements)
    assert reconstructed.shape == (10,)
