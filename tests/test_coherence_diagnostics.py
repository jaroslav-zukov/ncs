import matplotlib
import numpy as np
import pandas as pd
import pytest

import ncs.coherence_diagnostics as coherence_diagnostics_module
from ncs.coherence_diagnostics import (
    coherence_heatmap,
    compute_gram_matrix,
    empirical_rip_constant,
    flip_test,
    local_coherence_matrix,
    mutual_coherence,
    optimal_multilevel_allocation,
    phase_transition_grid,
)
from ncs.measurement_module import create_subsampling_operator
from ncs.wavelet_module import forward_transform, inverse_transform

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
        seed=123,
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
        seed=123,
    )

    expected_keys = {"delta_k", "mean_ratio", "std_ratio", "min_ratio", "max_ratio"}
    assert set(stats.keys()) == expected_keys
    assert all(np.isfinite(value) for value in stats.values())


def test_phase_transition_grid_contract():
    n = 16

    grid = phase_transition_grid(
        measurement_mode="subsampling",
        n=n,
        wavelet="haar",
        m_values=[8],
        k_values=[1, 2],
        n_trials=2,
    )

    assert isinstance(grid, pd.DataFrame)
    assert list(grid.columns) == ["m", "k", "recovery_probability"]
    assert len(grid) == 2
    assert np.all((grid["recovery_probability"] >= 0.0) & (grid["recovery_probability"] <= 1.0))


def test_phase_transition_grid_threads_measurement_mode(monkeypatch):
    n = 16
    called_modes: list[str] = []

    def fake_measure_and_reconstruct(
        measurement_mode: str,
        m: int,
        reconstruction_mode: str,
        coeffs_x,
        target_tree_sparsity: int,
        seed: int,
    ):
        called_modes.append(measurement_mode)
        return coeffs_x

    monkeypatch.setattr(
        coherence_diagnostics_module,
        "measure_and_reconstruct",
        fake_measure_and_reconstruct,
    )

    grid = phase_transition_grid(
        measurement_mode="random_modulation",
        n=n,
        wavelet="haar",
        m_values=[8],
        k_values=[1, 2],
        n_trials=2,
    )

    assert len(grid) == 2
    assert len(called_modes) == 4
    assert set(called_modes) == {"random_modulation"}


def test_local_coherence_matrix_returns_metadata():
    n = 16
    G = np.random.default_rng(0).standard_normal((8, n))

    local_mu, metadata = local_coherence_matrix(G=G, n=n, wavelet="haar")

    assert local_mu.shape[0] == local_mu.shape[1]
    assert local_mu.shape[0] == len(metadata["band_boundaries"])
    assert local_mu.shape[1] == len(metadata["scale_boundaries"])
    assert np.all(np.isfinite(local_mu))

    bands = metadata["band_boundaries"]
    starts = [start for start, _ in bands]
    ends = [end for _, end in bands]

    assert starts[0] == 0
    assert ends[-1] == G.shape[0]
    assert np.all(np.diff(starts) >= 0)
    assert np.all(np.diff(ends) >= 0)
    assert all(start <= end for start, end in bands)
    assert all(ends[idx] == starts[idx + 1] for idx in range(len(bands) - 1))


def test_optimal_multilevel_allocation_sums_to_total_budget():
    n = 16
    coeffs = forward_transform(np.zeros(n), "haar")
    local_sparsities = np.arange(1, coeffs.max_level + 2, dtype=float)

    allocation = optimal_multilevel_allocation(
        local_sparsities=local_sparsities,
        n=n,
        wavelet="haar",
        total_m=20,
    )

    assert allocation.shape == local_sparsities.shape
    assert np.sum(allocation) == 20
    assert np.all(allocation >= 0)


def test_optimal_multilevel_allocation_returns_normalized_weights_without_budget():
    n = 16
    coeffs = forward_transform(np.zeros(n), "haar")
    local_sparsities = np.arange(1, coeffs.max_level + 2, dtype=float)

    allocation = optimal_multilevel_allocation(
        local_sparsities=local_sparsities,
        n=n,
        wavelet="haar",
        total_m=None,
    )

    assert allocation.shape == local_sparsities.shape
    assert np.all(np.isfinite(allocation))
    assert np.all(allocation >= 0.0)
    assert np.sum(allocation) == pytest.approx(1.0)


def test_optimal_multilevel_allocation_uses_coif9_theorem_constants():
    n = 1024
    coeffs = forward_transform(np.zeros(n), "coif9")
    level_count = coeffs.max_level + 1
    local_sparsities = np.arange(1, level_count + 1, dtype=float)

    scale_sizes = np.array([len(group) for group in coeffs.coeff_groups], dtype=float)
    scale_ends = np.cumsum(scale_sizes)
    scale_starts = np.concatenate(([0.0], scale_ends[:-1]))
    n_prev = np.maximum(scale_starts, 1.0)
    prefactor = (scale_ends - n_prev) / n_prev

    alpha = 3.2
    nu = 18.0
    expected_weights = np.zeros(level_count, dtype=float)
    for level_k in range(level_count):
        left = max(0, level_k - 1)
        right = min(level_count - 1, level_k + 1)
        neighbor_max = float(np.max(local_sparsities[left : right + 1]))

        coarse_to_fine = 0.0
        for level_l in range(level_k):
            coarse_to_fine += float(
                local_sparsities[level_l] * 2.0 ** (-(alpha - 0.5) * (level_k - level_l))
            )

        fine_to_coarse = 0.0
        for level_l in range(level_k + 1, level_count):
            fine_to_coarse += float(local_sparsities[level_l] * 2.0 ** (-nu * (level_l - level_k)))

        expected_weights[level_k] = prefactor[level_k] * (
            neighbor_max + coarse_to_fine + fine_to_coarse
        )

    expected_allocation = expected_weights / np.sum(expected_weights)

    allocation = optimal_multilevel_allocation(
        local_sparsities=local_sparsities,
        n=n,
        wavelet="coif9",
        total_m=None,
    )

    assert np.allclose(allocation, expected_allocation)


def test_flip_test_returns_expected_keys():
    n = 16

    def identity(signal: np.ndarray) -> np.ndarray:
        return signal

    def identity_adj(signal: np.ndarray) -> np.ndarray:
        return signal

    result = flip_test(
        measure_op=(identity, identity_adj),
        n=n,
        wavelet="haar",
        k=2,
        n_trials=2,
    )

    assert set(result.keys()) == {"mse_structured", "mse_flipped", "ratio"}
    assert np.isfinite(result["mse_structured"])
    assert np.isfinite(result["mse_flipped"])
    assert result["ratio"] >= 0 or np.isinf(result["ratio"])
