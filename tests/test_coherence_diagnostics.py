import matplotlib
import numpy as np
import pandas as pd
import pytest

from ncs.coherence_diagnostics import (
    coherence_heatmap,
    compute_gram_matrix,
    empirical_rip_constant,
    mutual_coherence,
    phase_transition_grid,
)
from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.measurement_module import create_subsampling_operator
from ncs.wavelet_module import forward_transform

matplotlib.use("Agg")


def test_compute_gram_matrix_shape_and_finite_values():
    n = 8
    measure_op = create_subsampling_operator(n=n, m=4, seed=123)

    G = compute_gram_matrix(measure_op=measure_op, n=n, wavelet="haar")

    assert G.shape == (4, n)
    assert np.all(np.isfinite(G))


def test_mutual_coherence_hadamard_unit_coherence():
    G = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=float,
    ) / 2.0

    mu = mutual_coherence(G)

    assert mu == pytest.approx(1.0)


def test_mutual_coherence_raises_on_zero_column():
    G = np.array([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="zero-norm"):
        mutual_coherence(G)


def test_coherence_heatmap_returns_figure_with_image():
    G = np.random.default_rng(0).standard_normal((6, 8))

    fig = coherence_heatmap(G)

    assert hasattr(fig, "axes")
    assert len(fig.axes) >= 1
    assert len(fig.axes[0].images) >= 1


def test_empirical_rip_constant_contract_tree_sparse():
    n = 16

    def identity_measure(signal: np.ndarray) -> np.ndarray:
        return signal

    stats = empirical_rip_constant(
        measure_op=identity_measure,
        n=n,
        wavelet="haar",
        k=3,
        n_trials=20,
        tree_sparse=True,
    )

    expected_keys = {"delta_k", "mean_ratio", "std_ratio", "min_ratio", "max_ratio"}
    assert set(stats.keys()) == expected_keys
    assert all(np.isfinite(value) for value in stats.values())


def test_empirical_rip_constant_contract_uniform_sparse():
    n = 16

    def identity_measure(signal: np.ndarray) -> np.ndarray:
        return signal

    stats = empirical_rip_constant(
        measure_op=identity_measure,
        n=n,
        wavelet="haar",
        k=3,
        n_trials=20,
        tree_sparse=False,
    )

    expected_keys = {"delta_k", "mean_ratio", "std_ratio", "min_ratio", "max_ratio"}
    assert set(stats.keys()) == expected_keys
    assert all(np.isfinite(value) for value in stats.values())


def test_phase_transition_grid_contract():
    n = 8

    def factory(signal_len: int, m: int, seed: int | None = None):
        return create_subsampling_operator(signal_len, m, seed)

    grid = phase_transition_grid(
        measure_op_factory=factory,
        n=n,
        wavelet="haar",
        m_values=[4],
        k_values=[1, 2],
        n_trials=2,
    )

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == ["m", "k", "recovery_probability"]
    assert len(grid) == 2
    assert np.all((grid["recovery_probability"] >= 0.0) & (grid["recovery_probability"] <= 1.0))


def test_measure_and_reconstruct_backwards_compatibility_without_factory(mocker):
    coeffs_x = forward_transform(np.zeros(8), "haar")
    mock_create_operator = mocker.patch(
        "ncs.compressed_sensing_module.create_measurement_operator",
        return_value=create_subsampling_operator(8, 4, seed=7),
    )
    mock_reconstruct = mocker.patch(
        "ncs.compressed_sensing_module.reconstruct",
        side_effect=lambda **kwargs: kwargs["x_init"],
    )

    result = measure_and_reconstruct(
        measurement_mode="subsampling",
        m=4,
        reconstruction_mode="CoSaMP",
        coeffs_x=coeffs_x,
        target_tree_sparsity=2,
        seed=11,
    )

    mock_create_operator.assert_called_once_with("subsampling", 8, 4, 11)
    mock_reconstruct.assert_called_once()
    assert result.n == coeffs_x.n


def test_measure_and_reconstruct_uses_factory_override(mocker):
    coeffs_x = forward_transform(np.zeros(8), "haar")
    mock_create_operator = mocker.patch(
        "ncs.compressed_sensing_module.create_measurement_operator"
    )
    mock_reconstruct = mocker.patch(
        "ncs.compressed_sensing_module.reconstruct",
        side_effect=lambda **kwargs: kwargs["x_init"],
    )

    def factory(signal_len: int, m: int, seed: int | None = None):
        return create_subsampling_operator(signal_len, m, seed)

    result = measure_and_reconstruct(
        measurement_mode="subsampling",
        m=4,
        reconstruction_mode="CoSaMP",
        coeffs_x=coeffs_x,
        target_tree_sparsity=2,
        seed=11,
        measurement_op_factory=factory,
    )

    mock_create_operator.assert_not_called()
    mock_reconstruct.assert_called_once()
    assert result.n == coeffs_x.n

