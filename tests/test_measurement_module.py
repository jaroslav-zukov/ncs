import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ncs.measurement_module import (
    create_hadamard_multilevel_operator,
    create_hadamard_operator,
    create_fourier_subsampling_operator,
    create_gaussian_operator,
    create_measurement_operators,
    create_random_modulation_operator,
    create_subsampling_operator,
)


def test_create_subsampling_operator():
    subsample_op, upsample_op, pseudo_inverse_op = create_subsampling_operator(10, 5, 123)

    subsampled_signal = subsample_op(np.arange(10, dtype=float))
    # Values at selected indices scaled by sqrt(n/m) = sqrt(2)
    assert_allclose(subsampled_signal, np.array([0, 4, 7, 8, 9]) * np.sqrt(10 / 5))

    upsampled_signal = upsample_op([1, 2, 3, 4, 5])
    # Scattered values scaled by sqrt(n/m) = sqrt(2)
    assert_allclose(upsampled_signal, np.array([1, 0, 0, 0, 2, 0, 0, 3, 4, 5]) * np.sqrt(10 / 5))

    # Verifying subsample ∘ pseudo_inverse = I
    up_subsampled_signal = subsample_op(pseudo_inverse_op([1, 2, 3, 4, 5]))
    assert_allclose(up_subsampled_signal, [1, 2, 3, 4, 5])

    # Verify adjoint property: <Ax, y> = <x, A^T y>
    # For any vectors x and y holds:
    # measurement_op(x) · y == x · adjoint_op(y)
    signal = np.arange(10, dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    left_side = np.dot(subsample_op(signal), y)
    right_side = np.dot(signal, upsample_op(y))
    assert_allclose(left_side, right_side)


def test_create_gaussian_operator():
    measure_op, adjoint_op, _ = create_gaussian_operator(10, 5, 123)

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
    subsample_op, upsample_op, _ = create_measurement_operators(
        "subsampling", n=10, m=5, seed=123
    )
    signal = np.arange(10, dtype=float)
    subsampled = subsample_op(signal)
    # Values at selected indices scaled by sqrt(n/m) = sqrt(2)
    assert_allclose(subsampled, np.array([0, 4, 7, 8, 9]) * np.sqrt(10 / 5))

    upsampled = upsample_op(subsampled)
    assert upsampled.shape == (10,)

    # Test gaussian operators
    measure_op, adjoint_op, _ = create_measurement_operators(
        "gaussian", n=10, m=5, seed=123
    )
    measurements = measure_op(signal)
    assert measurements.shape == (5,)
    assert_allclose(measurements, [1.7, 3.8, 12.5, 7.5, 1.7], atol=0.1)

    reconstructed = adjoint_op(measurements)
    assert reconstructed.shape == (10,)


def test_random_modulation_operator_shape():
    n, m = 64, 20
    measure_op, adjoint_op, pinv_op = create_random_modulation_operator(n, m, seed=42)

    signal = np.random.default_rng(0).standard_normal(n)
    y = np.random.default_rng(1).standard_normal(m)

    assert measure_op(signal).shape == (m,)
    assert adjoint_op(y).shape == (n,)
    assert pinv_op(y).shape == (n,)


def test_random_modulation_adjoint_correctness():
    """Verify ⟨Φx, y⟩ = ⟨x, Φᵀy⟩ for random vectors."""
    n, m = 64, 20
    measure_op, adjoint_op, _ = create_random_modulation_operator(n, m, seed=99)

    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    y = rng.standard_normal(m)

    left = np.dot(measure_op(x), y)
    right = np.dot(x, adjoint_op(y))

    assert_allclose(left, right, rtol=1e-10)


def test_random_modulation_chipping_deterministic():
    """Same seed must produce identical measurements."""
    n, m = 32, 10
    measure1, _, _ = create_random_modulation_operator(n, m, seed=123)
    measure2, _, _ = create_random_modulation_operator(n, m, seed=123)

    signal = np.arange(n, dtype=float)
    assert_allclose(measure1(signal), measure2(signal))


def test_fourier_subsampling_operator_shape():
    n, m = 64, 20
    measure_op, adjoint_op, pseudo_inverse_op = create_fourier_subsampling_operator(n, m, seed=42)

    signal = np.random.default_rng(0).standard_normal(n)
    y = measure_op(signal)

    # Forward op: R^n → C^m
    assert y.shape == (m,)
    assert np.iscomplexobj(y)

    # Adjoint op: C^m → R^n
    x_adj = adjoint_op(y)
    assert x_adj.shape == (n,)
    assert np.isrealobj(x_adj)

    # Pseudo-inverse op: C^m → R^n
    x_pinv = pseudo_inverse_op(y)
    assert x_pinv.shape == (n,)


def test_fourier_subsampling_adjoint_correctness():
    """Verify the adjoint property for Phi: R^n to C^m.

    The real-valued adjoint Phi^T: C^m to R^n satisfies:
        Re(<Phi x, y>_C) = <x, Phi^T y>_R
    i.e. Re(sum_i (Phi x)_i * conj(y_i)) = sum_j x_j * (Phi^T y)_j
    """
    n, m = 128, 40
    measure_op, adjoint_op, _ = create_fourier_subsampling_operator(n, m, seed=7)

    rng = np.random.default_rng(99)
    x = rng.standard_normal(n)
    y = rng.standard_normal(m) + 1j * rng.standard_normal(m)

    # Re(<Phi x, y>_C) = Re(sum_i (Phi x)[i] * conj(y[i]))
    left_side = np.real(np.dot(measure_op(x), np.conj(y)))

    # <x, Phi^T y>_R  (adjoint_op returns a real array)
    right_side = np.dot(x, adjoint_op(y))

    assert_allclose(left_side, right_side, rtol=1e-10)


def test_fourier_subsampling_unitary_pseudoinverse():
    """Verify adjoint == pseudo_inverse (consequence of unitary rfft with norm='ortho')."""
    n, m = 64, 15
    _, adjoint_op, pseudo_inverse_op = create_fourier_subsampling_operator(n, m, seed=13)

    rng = np.random.default_rng(55)
    y = rng.standard_normal(m) + 1j * rng.standard_normal(m)

    assert_allclose(adjoint_op(y), pseudo_inverse_op(y), rtol=1e-12)


def test_fourier_subsampling_m_too_large_raises():
    """Raise ValueError when m exceeds n//2+1 unique rfft frequencies."""
    n = 32
    n_freqs = n // 2 + 1  # = 17
    with pytest.raises(ValueError, match="exceeds number of unique frequencies"):
        create_fourier_subsampling_operator(n, m=n_freqs + 1)


def test_create_measurement_operators_fourier_subsampling():
    n, m = 64, 20
    measure_op, adjoint_op, pseudo_inverse_op = create_measurement_operators(
        "fourier_subsampling", n=n, m=m, seed=42
    )
    signal = np.random.default_rng(0).standard_normal(n)
    y = measure_op(signal)
    assert y.shape == (m,)
    assert np.iscomplexobj(y)

    x_adj = adjoint_op(y)
    assert x_adj.shape == (n,)


def test_hadamard_operator_shape_and_adjoint():
    n, m = 32, 10
    measure_op, adjoint_op, pseudo_inverse_op = create_hadamard_operator(n, m, seed=7)

    x = np.random.default_rng(3).standard_normal(n)
    y = np.random.default_rng(4).standard_normal(m)

    assert measure_op(x).shape == (m,)
    assert adjoint_op(y).shape == (n,)
    assert pseudo_inverse_op(y).shape == (n,)

    left = np.dot(measure_op(x), y)
    right = np.dot(x, adjoint_op(y))
    assert_allclose(left, right, rtol=1e-10)


def test_hadamard_operator_sequency_dc_first():
    n = 8
    measure_op, _, _ = create_hadamard_operator(n, n, seed=5)

    coeffs = measure_op(np.ones(n))
    assert coeffs[0] == pytest.approx(np.sqrt(n), rel=1e-10, abs=1e-10)
    assert_allclose(coeffs[1:], np.zeros(n - 1), atol=1e-10)


def test_hadamard_multilevel_operator_metadata_and_shape():
    n, m = 64, 20
    measure_op, adjoint_op, pseudo_inverse_op, metadata = create_hadamard_multilevel_operator(
        n=n,
        m=m,
        wavelet="haar",
        seed=11,
    )

    assert measure_op(np.ones(n)).shape == (m,)
    assert adjoint_op(np.ones(m)).shape == (n,)
    assert pseudo_inverse_op(np.ones(m)).shape == (n,)

    assert len(metadata["allocation"]) == int(np.log2(n))
    assert int(np.sum(metadata["allocation"])) == m
    assert len(metadata["band_boundaries"]) == int(np.log2(n))


def test_measurement_dispatcher_supports_hadamard_modes():
    n, m = 64, 20

    hadamard_ops = create_measurement_operators("hadamard", n=n, m=m, seed=17)
    multilevel_ops = create_measurement_operators("hadamard_multilevel", n=n, m=m, seed=17)

    assert len(hadamard_ops) == 3
    assert len(multilevel_ops) == 3


def test_hadamard_multilevel_coif9_tracks_local_sparsity_profile():
    n, m = 64, 24
    local_sparsities = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    _, _, _, metadata = create_hadamard_multilevel_operator(
        n=n,
        m=m,
        wavelet="coif9",
        local_sparsities=local_sparsities,
        seed=19,
    )

    allocation = metadata["allocation"]
    assert int(np.sum(allocation)) == m
    assert allocation[-1] >= allocation[0]
    assert allocation[-2] >= allocation[1]
