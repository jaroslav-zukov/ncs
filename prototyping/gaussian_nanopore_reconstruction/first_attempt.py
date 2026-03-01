from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from ncs import load_signal
from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs


def from_flat(array):
    return WtCoeffs.from_flat_coeffs(array, 64, 7, "coif9")


def flat_support(array):
    return np.nonzero(array)[0]


def tree_cosamp(phi, y, target_s):
    n = phi.shape[1]
    x_hat = np.zeros(n)
    r = y.copy()

    for _ in range(50):
        e_coeffs = from_flat(phi.T @ r)

        omega_e = list(tree_projection(e_coeffs, 2 * target_s).support)
        current_support = flat_support(x_hat)

        t = np.union1d(current_support, omega_e).astype(int)

        phi_t = phi[:, t]
        b_t, _, _, _ = np.linalg.lstsq(phi_t, y, rcond=None)

        b = np.zeros(n)
        b[t] = b_t

        b_coeffs = from_flat(b)

        x_hat = tree_projection(b_coeffs, target_s).flat_coeffs

        r = y - (phi @ x_hat)

        if np.linalg.norm(r) < 1e-10:
            break

    return x_hat


def generate_reconstruction_data(m_values, signal, n, target_s):
    results = []

    for m in tqdm(m_values):
        original_x_coeffs = forward_transform(signal, "coif9")

        # generating measurement matrix
        rng = np.random.default_rng()
        phi = rng.normal(0, 1 / np.sqrt(m), size=(m, n))

        y = phi @ original_x_coeffs.flat_coeffs

        x_hat_flat = tree_cosamp(phi, y, target_s)

        x_hat_coeffs = from_flat(x_hat_flat)
        missed_support_len = len(original_x_coeffs.support - x_hat_coeffs.support)
        z_hat = inverse_transform(x_hat_coeffs)

        norm_signal = np.linalg.norm(signal)
        relative_error = np.linalg.norm(signal - z_hat) / norm_signal

        results.append(
            {
                "m": m,
                "relative_error": relative_error,
                "missed_support": missed_support_len,
            }
        )

    return pd.DataFrame(results)


def plot_squiggle_reconstruction(m_values, signal, n, target_s):
    df = generate_reconstruction_data(m_values, signal, n, target_s)
    params = {
        "N": n,
        "Tree sparsity": target_s,
    }
    param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot for individual trials
    sns.stripplot(
        data=df,
        x="m",
        y="relative_error",
        hue="missed_support",
        jitter=0.2,
        alpha=0.7,
        palette="RdYlGn_r",
        native_scale=True,
        ax=ax,
    )

    # Mean line
    sns.lineplot(
        data=df,
        x="m",
        y="relative_error",
        estimator="mean",
        color="blue",
        linewidth=2,
        errorbar=None,
        ax=ax,
        label="Mean Relative Error",
    )

    ax.set_xlabel(
        f"Number of Measurements (m {m_values.min()} - {m_values.max()})", fontsize=12
    )
    ax.set_ylabel("Relative Error", fontsize=12)
    ax.set_title(
        "Sparse Signal Reconstruction: Relative Error vs Number of Measurements",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Missed Support", fontsize=10, loc="upper right")

    at = AnchoredText(
        s=param_str,
        loc="upper right",
        bbox_to_anchor=(1, 0.73),
        bbox_transform=ax.transAxes,
        frameon=True,
    )
    ax.add_artist(at)

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"squiggle_gaussian_reconstruction_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.show()


def main():
    print("Nanopore signal reconstruction attempt")
    # Choose set parameters (n,s, signal_count, reconstruction attempts)
    power = 13
    n = 2**power
    signal = load_signal(power, 1)[0]
    m_values = np.linspace(100, 2000, 30, dtype=int)

    plot_squiggle_reconstruction(m_values, signal, n, 800)


if __name__ == "__main__":
    main()
