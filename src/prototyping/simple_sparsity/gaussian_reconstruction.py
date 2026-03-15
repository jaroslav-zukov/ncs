from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm


def generate_sparse_x(n, s):
    signal = np.zeros(n)
    indices = np.sort(np.random.choice(n, s, replace=False))
    # signal[indices] = np.random.random(s) # float from 0 to 1
    signal[indices] = np.random.randint(-300, 300, size=s) # integer from -300 to 300
    return signal


def sparse_projection(array, sparsity):
    result = np.zeros_like(array)
    indices = np.argsort(np.abs(array))[-sparsity:]
    result[indices] = array[indices]
    return result


def support(array):
    return set(np.nonzero(array)[0])


def classical_cosamp(phi, y, target_s):
    n = phi.shape[1]
    x_hat = np.zeros(n)
    r = y.copy()

    for _ in range(50):
        e = phi.T @ r

        omega_e = np.nonzero(sparse_projection(e, 2 * target_s))[0]

        current_support = np.nonzero(x_hat)[0]
        t = np.union1d(current_support, omega_e).astype(int)

        phi_t = phi[:, t]
        # what x produces minimal square error with y, when measured
        b_t, _, _, _ = np.linalg.lstsq(phi_t, y, rcond=None)

        b = np.zeros(n)
        b[t] = b_t

        x_hat = sparse_projection(array=b, sparsity=target_s)

        r = y - (phi @ x_hat)

        if np.linalg.norm(r) < 1e-10:
            break

    return x_hat


def generate_sparse_signal_reconstruction_data(
    n,
    s,
    signal_count,
    reconstruction_attempts,
    m_values,
):
    results = []

    for m in tqdm(m_values, desc="Measuring and reconstructing"):
        for _ in range(signal_count):
            sparse_signal = generate_sparse_x(n, s)

            rng = np.random.default_rng()
            phi = rng.normal(0, 1 / np.sqrt(m), size=(m, n))

            y = phi @ sparse_signal

            for _ in range(reconstruction_attempts):
                x_hat = classical_cosamp(
                    phi=phi,
                    y=y,
                    target_s=s,
                )

                missed_support_len = len(support(sparse_signal) - support(x_hat))

                norm_signal = np.linalg.norm(sparse_signal)

                relative_error = np.linalg.norm(sparse_signal - x_hat) / norm_signal

                results.append(
                    {
                        "m": m,
                        "relative_error": relative_error,
                        "missed_support": missed_support_len,
                    }
                )

    df = pd.DataFrame(results)

    params = {
        "N": n,
        "Sparsity": s,
        "Signals": signal_count,
        "Reconstructions per signal": reconstruction_attempts,
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
    filename = f"figures/sparse_gaussian_reconstruction_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    plt.show()


def main():
    print("Gaussian Reconstruction (CoSaMP)")
    m_values = np.linspace(20, 150, 20).astype(int)

    generate_sparse_signal_reconstruction_data(
        n=1000,
        s=10,
        signal_count=10,
        reconstruction_attempts=2,
        m_values=m_values,
    )


if __name__ == "__main__":
    main()
