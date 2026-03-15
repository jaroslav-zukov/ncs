"""
Noisy sparse signal reconstruction experiment (script version of notebook 1.1).

This script reconstructs noisy tree-sparse signals and saves one summary plot
of mean signal-domain MSE vs number of measurements.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.config import FIGURES_DIR
from ncs.sparse_signal_generator import add_noise_to_coeffs, generate_tree_sparse_coeffs
from ncs.wavelet_module import inverse_transform


def main() -> None:
    n_power = 10
    wavelet = "haar"
    tree_sparsity = 50
    noise_epsilon = 0.05
    measurement_mode = "gaussian"
    reconstruction_mode = "CoSaMP"
    m_values = np.arange(120, 481, 40)
    n_trials = 4

    clean_coeffs = generate_tree_sparse_coeffs(
        power=n_power,
        count=n_trials,
        tree_sparsity=tree_sparsity,
        wavelet=wavelet,
        seed=1,
    )
    noisy_coeffs = add_noise_to_coeffs(
        tree_sparse_signals=clean_coeffs,
        noise_epsilon=noise_epsilon,
        noise_mode="gaussian",
        seed=1,
    )

    mean_mse = []
    std_mse = []
    for m in m_values:
        mse_values = []
        for idx, (clean, noisy) in enumerate(zip(clean_coeffs, noisy_coeffs)):
            x_hat = measure_and_reconstruct(
                measurement_mode=measurement_mode,
                m=int(m),
                reconstruction_mode=reconstruction_mode,
                coeffs_x=noisy,
                target_tree_sparsity=tree_sparsity,
                seed=200 + idx,
            )
            mse = np.mean((inverse_transform(clean) - inverse_transform(x_hat)) ** 2)
            mse_values.append(float(mse))
        mean_mse.append(float(np.mean(mse_values)))
        std_mse.append(float(np.std(mse_values)))

    output_dir = FIGURES_DIR / "notebook_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "1_1_noisy_signal_reconstruction.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(m_values, mean_mse, yerr=std_mse, marker="o", capsize=4, color="tab:red")
    ax.set_xlabel("Number of measurements (m)")
    ax.set_ylabel("Signal-domain MSE")
    ax.set_title("Noisy Tree-Sparse Reconstruction vs Measurements")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

