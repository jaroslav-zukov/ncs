"""
Sparse signal reconstruction experiment (script version of notebook 1.0).

This script runs a compact reconstruction sweep over measurement counts and
stores exactly one image summarising mean signal-domain MSE.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.config import FIGURES_DIR
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs
from ncs.wavelet_module import inverse_transform


def main() -> None:
    n_power = 10
    wavelet = "haar"
    tree_sparsity = 20
    measurement_mode = "gaussian"
    reconstruction_mode = "CoSaMP"
    m_values = np.arange(80, 401, 40)
    n_trials = 4

    coeffs = generate_tree_sparse_coeffs(
        power=n_power,
        count=n_trials,
        tree_sparsity=tree_sparsity,
        wavelet=wavelet,
        seed=0,
    )

    mean_mse = []
    std_mse = []
    for m in m_values:
        trial_mse = []
        for idx, coeff in enumerate(coeffs):
            x_hat = measure_and_reconstruct(
                measurement_mode=measurement_mode,
                m=int(m),
                reconstruction_mode=reconstruction_mode,
                coeffs_x=coeff,
                target_tree_sparsity=tree_sparsity,
                seed=100 + idx,
            )
            mse = np.mean((inverse_transform(coeff) - inverse_transform(x_hat)) ** 2)
            trial_mse.append(float(mse))

        mean_mse.append(float(np.mean(trial_mse)))
        std_mse.append(float(np.std(trial_mse)))

    output_dir = FIGURES_DIR / "notebook_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "1_0_sparse_signal_reconstruction.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(m_values, mean_mse, yerr=std_mse, marker="o", capsize=4)
    ax.set_xlabel("Number of measurements (m)")
    ax.set_ylabel("Signal-domain MSE")
    ax.set_title("Sparse Tree Reconstruction vs Measurements")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

