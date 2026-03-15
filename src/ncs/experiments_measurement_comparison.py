from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ncs.coherence_diagnostics import compute_gram_matrix
from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.config import FIGURES_DIR, PROCESSED_DATA_DIR
from ncs.measurement_module import create_measurement_operator
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs
from ncs.wavelet_module import inverse_transform


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_output_file(output_path: Path | str, suffix: str, stem_prefix: str) -> Path:
    path = Path(output_path)
    if path.suffix == suffix:
        resolved = path
    else:
        resolved = path / f"{stem_prefix}_{_timestamp()}{suffix}"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _support_recovery_rate(true_support: set[int], recovered_support: set[int]) -> float:
    if len(true_support) == 0:
        return 1.0
    return float(len(true_support & recovered_support) / len(true_support))


def run_measurement_comparison_experiment(
    *,
    n: int = 4096,
    tree_sparsity: int = 100,
    wavelet: str = "coif9",
    signal_count: int = 20,
    measurement_modes: list[str] | None = None,
    m_values: np.ndarray | None = None,
    reconstruction_mode: str = "CoSaMP",
    seed: int = 0,
    exact_recovery_mse_threshold: float = 1e-6,
) -> pd.DataFrame:
    if not _is_power_of_two(n):
        raise ValueError("n must be a power of 2")
    if signal_count < 1:
        raise ValueError("signal_count must be >= 1")
    if tree_sparsity < 1:
        raise ValueError("tree_sparsity must be >= 1")

    if measurement_modes is None:
        measurement_modes = [
            "gaussian",
            "fourier_subsampling",
            "random_modulation",
            "hadamard",
            "hadamard_multilevel",
        ]

    if m_values is None:
        m_values = np.linspace(150, 1500, 20).astype(int)
    m_values = np.asarray(m_values, dtype=int)

    power = int(np.log2(n))
    sparse_coeffs = generate_tree_sparse_coeffs(
        power=power,
        count=signal_count,
        tree_sparsity=tree_sparsity,
        wavelet=wavelet,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for measurement_mode in measurement_modes:
        for m in m_values:
            m_int = int(m)
            for signal_index, coeffs_x in enumerate(sparse_coeffs):
                operator_seed = int(rng.integers(0, np.iinfo(np.int32).max))

                started_at = perf_counter()
                reconstructed = measure_and_reconstruct(
                    measurement_mode=measurement_mode,
                    m=m_int,
                    reconstruction_mode=reconstruction_mode,
                    coeffs_x=coeffs_x,
                    target_tree_sparsity=tree_sparsity,
                    seed=operator_seed,
                )
                elapsed = perf_counter() - started_at

                signal_true = inverse_transform(coeffs_x)
                signal_reconstructed = inverse_transform(reconstructed)
                mse = float(np.mean((signal_true - signal_reconstructed) ** 2))
                support_recall = _support_recovery_rate(coeffs_x.support, reconstructed.support)

                records.append(
                    {
                        "measurement_mode": measurement_mode,
                        "m": m_int,
                        "signal_index": int(signal_index),
                        "m_over_k": float(m_int / tree_sparsity),
                        "mse": mse,
                        "support_recovery_rate": support_recall,
                        "exact_recovery": int(mse < exact_recovery_mse_threshold),
                        "wall_clock_time_s": float(elapsed),
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "measurement_mode",
            "m",
            "signal_index",
            "m_over_k",
            "mse",
            "support_recovery_rate",
            "exact_recovery",
            "wall_clock_time_s",
        ],
    )


def save_measurement_comparison_results(df: pd.DataFrame, output_path: Path | str) -> Path:
    csv_path = _resolve_output_file(
        output_path=output_path,
        suffix=".csv",
        stem_prefix="measurement_comparison",
    )
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_mse_vs_mk(df: pd.DataFrame, output_path: Path | str, tree_sparsity: int) -> Path:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x="m_over_k",
        y="mse",
        hue="measurement_mode",
        estimator="mean",
        errorbar=("ci", 95),
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("m / k")
    ax.set_ylabel("MSE")
    ax.set_title(f"Measurement Operator Comparison: MSE vs m/k (k={tree_sparsity})")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    figure_path = _resolve_output_file(
        output_path=output_path,
        suffix=".png",
        stem_prefix="measurement_comparison_mse_vs_mk",
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_phase_transition(
    df: pd.DataFrame,
    output_path: Path | str,
    tree_sparsity: int,
    threshold: float = 0.95,
) -> Path:
    grouped = (
        df.groupby(["measurement_mode", "m_over_k"], as_index=False)["exact_recovery"]
        .mean()
        .rename(columns={"exact_recovery": "exact_recovery_probability"})
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=grouped,
        x="m_over_k",
        y="exact_recovery_probability",
        hue="measurement_mode",
        marker="o",
        ax=ax,
    )
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"P={threshold}")

    ax.set_xlabel("m / k")
    ax.set_ylabel("P(exact recovery)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Empirical Phase Transition vs m/k (k={tree_sparsity})")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    figure_path = _resolve_output_file(
        output_path=output_path,
        suffix=".png",
        stem_prefix="measurement_comparison_phase_transition",
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_operator_coherence_heatmaps(
    *,
    output_path: Path | str,
    measurement_modes: list[str] | None = None,
    n: int = 512,
    m: int = 256,
    wavelet: str = "coif9",
    seed: int = 0,
) -> Path:
    if measurement_modes is None:
        measurement_modes = [
            "gaussian",
            "fourier_subsampling",
            "random_modulation",
            "hadamard",
            "hadamard_multilevel",
        ]

    n_modes = len(measurement_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), squeeze=False)
    axes_flat = axes.flatten()

    for mode_index, measurement_mode in enumerate(measurement_modes):
        operator_seed = seed + mode_index
        measure_op = create_measurement_operator(
            measurement_mode=measurement_mode,
            n=n,
            m=m,
            seed=operator_seed,
        )
        gram = compute_gram_matrix(measure_op=measure_op, n=n, wavelet=wavelet)

        ax = axes_flat[mode_index]
        im = ax.imshow(np.abs(gram) ** 2, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(measurement_mode)
        ax.set_xlabel("Coefficient index")
        if mode_index == 0:
            ax.set_ylabel("Measurement index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Coherence heatmaps | n={n}, m={m}, wavelet={wavelet}")
    fig.tight_layout()

    figure_path = _resolve_output_file(
        output_path=output_path,
        suffix=".png",
        stem_prefix="measurement_comparison_coherence_heatmaps",
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main() -> None:
    measurement_modes = [
        "gaussian",
        "fourier_subsampling",
        "random_modulation",
        "hadamard",
        "hadamard_multilevel",
    ]
    m_values = np.linspace(150, 1500, 20).astype(int)

    df = run_measurement_comparison_experiment(
        n=4096,
        tree_sparsity=100,
        wavelet="coif9",
        signal_count=20,
        measurement_modes=measurement_modes,
        m_values=m_values,
        reconstruction_mode="CoSaMP",
        seed=0,
        exact_recovery_mse_threshold=1e-6,
    )

    csv_path = save_measurement_comparison_results(df, PROCESSED_DATA_DIR)
    mse_figure_path = plot_mse_vs_mk(df, FIGURES_DIR, tree_sparsity=100)
    phase_figure_path = plot_phase_transition(df, FIGURES_DIR, tree_sparsity=100, threshold=0.95)
    heatmap_path = plot_operator_coherence_heatmaps(
        output_path=FIGURES_DIR,
        measurement_modes=measurement_modes,
        n=512,
        m=256,
        wavelet="coif9",
        seed=0,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved MSE figure: {mse_figure_path}")
    print(f"Saved phase figure: {phase_figure_path}")
    print(f"Saved coherence heatmaps: {heatmap_path}")


if __name__ == "__main__":
    main()

