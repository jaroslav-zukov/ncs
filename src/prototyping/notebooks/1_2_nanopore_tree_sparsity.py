"""
Nanopore tree-sparsity profile (script version of notebook 1.2).

This script evaluates approximation quality of tree projection over sparsity
levels and writes one image with the MSE curve.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ncs import load_signal
from ncs.config import FIGURES_DIR
from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, inverse_transform


def _load_or_generate_signal(power: int) -> np.ndarray:
    try:
        return np.asarray(load_signal(power=power, count=1).iloc[0], dtype=float)
    except Exception:
        rng = np.random.default_rng(42)
        base = rng.standard_normal(2**power)
        return np.cumsum(base)


def main() -> None:
    n_power = 13
    wavelet = "coif9"
    signal = _load_or_generate_signal(n_power)
    wt_coeffs = forward_transform(signal, wavelet)

    sparsity_levels = np.array([50, 100, 200, 400, 800, 1200])
    mse_values = []
    for level in sparsity_levels:
        projected = tree_projection(wt_coeffs, int(level))
        reconstructed = inverse_transform(projected)
        mse_values.append(float(np.mean((signal - reconstructed) ** 2)))

    output_dir = FIGURES_DIR / "notebook_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "1_2_nanopore_tree_sparsity.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(sparsity_levels, mse_values, marker="o", color="tab:green")
    ax.set_xlabel("Tree sparsity k")
    ax.set_ylabel("Approximation MSE (log scale)")
    ax.set_title("Nanopore Signal: Tree-Projection Error vs Sparsity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

