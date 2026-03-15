import matplotlib
import numpy as np
import pandas as pd

import ncs.experiments_measurement_comparison as measurement_comparison
from ncs.wavelet_module import forward_transform
from ncs.wt_coeffs import WtCoeffs

matplotlib.use("Agg")


def _make_sparse_coeffs(
    *,
    n: int,
    wavelet: str,
    support: tuple[int, ...],
    values: tuple[float, ...],
) -> WtCoeffs:
    template = forward_transform(np.zeros(n), wavelet)
    flat = np.zeros(n)
    for index, value in zip(support, values, strict=True):
        flat[index] = value

    return WtCoeffs.from_flat_coeffs(
        flat_coeffs=flat,
        root_count=template.root_count,
        max_level=template.max_level,
        wavelet=wavelet,
    )


def test_run_measurement_comparison_experiment_contract(monkeypatch):
    coeffs_a = _make_sparse_coeffs(n=8, wavelet="haar", support=(1, 3), values=(2.0, -1.0))
    coeffs_b = _make_sparse_coeffs(n=8, wavelet="haar", support=(2, 4), values=(1.5, -0.5))

    def fake_generate_tree_sparse_coeffs(
        power: int,
        count: int,
        tree_sparsity: int,
        wavelet: str,
        seed: int,
    ):
        return [coeffs_a, coeffs_b]

    def fake_measure_and_reconstruct(
        measurement_mode: str,
        m: int,
        reconstruction_mode: str,
        coeffs_x: WtCoeffs,
        target_tree_sparsity: int,
        seed: int,
    ) -> WtCoeffs:
        first_supported = min(coeffs_x.support)
        return coeffs_x.on_support({first_supported})

    monkeypatch.setattr(
        measurement_comparison,
        "generate_tree_sparse_coeffs",
        fake_generate_tree_sparse_coeffs,
    )
    monkeypatch.setattr(
        measurement_comparison,
        "measure_and_reconstruct",
        fake_measure_and_reconstruct,
    )
    monkeypatch.setattr(
        measurement_comparison,
        "inverse_transform",
        lambda coeffs: coeffs.flat_coeffs,
    )

    modes = ["gaussian", "hadamard"]
    m_values = np.array([2, 4])
    result = measurement_comparison.run_measurement_comparison_experiment(
        n=8,
        tree_sparsity=2,
        wavelet="haar",
        signal_count=2,
        measurement_modes=modes,
        m_values=m_values,
        seed=0,
    )

    assert len(result) == len(modes) * len(m_values) * 2
    assert list(result.columns) == [
        "measurement_mode",
        "m",
        "signal_index",
        "m_over_k",
        "mse",
        "support_recovery_rate",
        "exact_recovery",
        "wall_clock_time_s",
    ]
    assert np.all((result["support_recovery_rate"] >= 0.0) & (result["support_recovery_rate"] <= 1.0))


def test_support_recovery_rate_uses_recall(monkeypatch):
    coeffs = _make_sparse_coeffs(n=8, wavelet="haar", support=(1, 3), values=(2.0, -1.0))

    monkeypatch.setattr(
        measurement_comparison,
        "generate_tree_sparse_coeffs",
        lambda power, count, tree_sparsity, wavelet, seed: [coeffs],
    )
    monkeypatch.setattr(
        measurement_comparison,
        "measure_and_reconstruct",
        lambda measurement_mode, m, reconstruction_mode, coeffs_x, target_tree_sparsity, seed: coeffs_x.on_support(
            {1}
        ),
    )
    monkeypatch.setattr(
        measurement_comparison,
        "inverse_transform",
        lambda wt_coeffs: wt_coeffs.flat_coeffs,
    )

    result = measurement_comparison.run_measurement_comparison_experiment(
        n=8,
        tree_sparsity=2,
        wavelet="haar",
        signal_count=1,
        measurement_modes=["gaussian"],
        m_values=np.array([2]),
        seed=1,
    )

    assert len(result) == 1
    assert result.iloc[0]["support_recovery_rate"] == 0.5


def test_save_measurement_comparison_results_roundtrip(tmp_path):
    df = pd.DataFrame(
        [
            {
                "measurement_mode": "gaussian",
                "m": 10,
                "signal_index": 0,
                "m_over_k": 2.0,
                "mse": 1e-3,
                "support_recovery_rate": 0.8,
                "exact_recovery": 0,
                "wall_clock_time_s": 0.05,
            }
        ]
    )

    output_file = tmp_path / "results.csv"
    saved_path = measurement_comparison.save_measurement_comparison_results(df, output_file)
    loaded = pd.read_csv(saved_path)

    assert saved_path.exists()
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)


def test_plot_functions_generate_files(tmp_path):
    df = pd.DataFrame(
        [
            {
                "measurement_mode": "gaussian",
                "m": 10,
                "signal_index": 0,
                "m_over_k": 1.0,
                "mse": 1e-3,
                "support_recovery_rate": 0.8,
                "exact_recovery": 0,
                "wall_clock_time_s": 0.05,
            },
            {
                "measurement_mode": "hadamard",
                "m": 20,
                "signal_index": 1,
                "m_over_k": 2.0,
                "mse": 5e-4,
                "support_recovery_rate": 0.9,
                "exact_recovery": 1,
                "wall_clock_time_s": 0.03,
            },
        ]
    )

    mse_path = measurement_comparison.plot_mse_vs_mk(
        df=df,
        output_path=tmp_path / "mse.png",
        tree_sparsity=10,
    )
    phase_path = measurement_comparison.plot_phase_transition(
        df=df,
        output_path=tmp_path / "phase.png",
        tree_sparsity=10,
    )

    assert mse_path.exists()
    assert phase_path.exists()


def test_plot_operator_coherence_heatmaps_generates_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        measurement_comparison,
        "create_measurement_operator",
        lambda measurement_mode, n, m, seed: (
            lambda signal: signal[:m],
            lambda meas: np.pad(meas, (0, n - len(meas))),
            lambda meas: np.pad(meas, (0, n - len(meas))),
        ),
    )
    monkeypatch.setattr(
        measurement_comparison,
        "compute_gram_matrix",
        lambda measure_op, n, wavelet: np.ones((4, n), dtype=float),
    )

    heatmap_path = measurement_comparison.plot_operator_coherence_heatmaps(
        output_path=tmp_path / "heatmaps.png",
        measurement_modes=["gaussian", "hadamard"],
        n=8,
        m=4,
        wavelet="haar",
        seed=0,
    )

    assert heatmap_path.exists()

