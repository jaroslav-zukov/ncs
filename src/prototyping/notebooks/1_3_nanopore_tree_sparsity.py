"""
Nanopore wavelet comparison (script version of notebook 1.3).

This script compares several orthogonal wavelets by tree-projection MSE and
stores one bar chart image.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ncs import load_signal
from ncs.config import FIGURES_DIR
from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, get_orthogonal_wavelets, inverse_transform


def _load_or_generate_signal(power: int) -> np.ndarray:
    try:
        return np.asarray(load_signal(power=power, count=1).iloc[0], dtype=float)
    except Exception:
        rng = np.random.default_rng(43)
        base = rng.standard_normal(2**power)
        return np.cumsum(base)


def main() -> None:
    n_power = 13
    signal = _load_or_generate_signal(n_power)
    tree_sparsity = 800

    candidates = [wavelet for wavelet in get_orthogonal_wavelets() if wavelet in {"haar", "db2", "db4", "sym4", "coif1", "coif3"}]
    if not candidates:
        candidates = ["haar"]

    scores = []
    for wavelet in candidates:
        coeffs = forward_transform(signal, wavelet)
        projected = tree_projection(coeffs, tree_sparsity)
        reconstructed = inverse_transform(projected)
        mse = float(np.mean((signal - reconstructed) ** 2))
        scores.append((wavelet, mse))

    scores.sort(key=lambda item: item[1])
    wavelets = [item[0] for item in scores]
    mse_values = [item[1] for item in scores]

    output_dir = FIGURES_DIR / "notebook_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "1_3_nanopore_tree_sparsity.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(wavelets, mse_values, color="tab:blue")
    ax.set_xlabel("Wavelet")
    ax.set_ylabel("Approximation MSE")
    ax.set_title("Wavelet Comparison on Nanopore Tree-Sparse Approximation")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

