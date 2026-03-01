from datetime import datetime
import argparse
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from ncs.measurement_module import create_subsampling_operator


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


@lru_cache(maxsize=None)
def coeff_array_spec(signal_length, wavelet):
    max_level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet).dec_len)
    template = pywt.wavedec(
        np.zeros(signal_length),
        wavelet=wavelet,
        mode="periodization",
        level=max_level,
    )
    coeff_arr, coeff_slices = pywt.coeffs_to_array(template)
    return coeff_arr.shape, coeff_slices, max_level


def flat_to_coeffs(flat_coeffs, shape, coeff_slices):
    if np.prod(shape) != len(flat_coeffs):
        raise ValueError(
            f"Flat coeffs length {len(flat_coeffs)} does not match wavelet shape {shape}."
        )
    coeff_arr = np.asarray(flat_coeffs).reshape(shape)
    return pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedec")


def coeffs_to_flat(coeffs):
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    return coeff_arr.ravel()


def project_on_support(array, support_indices):
    projected = np.zeros_like(array)
    projected[support_indices] = array[support_indices]
    return projected


def classical_cosamp(phi_op, phi_transpose_op, phi_pinv_op, y, target_s, n_coeffs):
    x_hat = np.zeros(n_coeffs)
    r = y.copy()

    for _ in range(50):
        e = phi_transpose_op(r)

        omega_e = np.nonzero(sparse_projection(e, 2 * target_s))[0]

        current_support = np.nonzero(x_hat)[0]
        t = np.union1d(current_support, omega_e).astype(int)

        b = project_on_support(phi_pinv_op(y), t)

        x_hat = sparse_projection(array=b, sparsity=target_s)

        r = y - phi_op(x_hat)

        if np.linalg.norm(r) < 1e-10:
            break

    return x_hat


def generate_sparse_signal_reconstruction_data(
    n,
    s,
    signal_count,
    reconstruction_attempts,
    m_values,
    show_plot=True,
    plot_example_m=500,
):
    results = []
    example_pair = None

    for m in tqdm(m_values, desc="Measuring and reconstructing"):
        for _ in range(signal_count):
            wavelet = "haar"
            subsample, transposed, pseudo_inverse = create_subsampling_operator(n, m)

            shape, coeff_slices, level = coeff_array_spec(n, wavelet)
            n_coeffs = int(np.prod(shape))
            if n_coeffs != n:
                raise ValueError(
                    f"Wavelet coefficient length ({n_coeffs}) != signal length ({n}). "
                    "Use a power-of-two signal length (e.g., 1024) for an orthogonal transform."
                )

            sparse_signal = generate_sparse_x(n_coeffs, s)

            def phi_op(x_flat):
                coeffs = flat_to_coeffs(x_flat, shape, coeff_slices)
                signal = pywt.waverec(coeffs, wavelet=wavelet, mode="periodization")
                return subsample(signal)

            def phi_transpose_op(measurements):
                upsampled = transposed(measurements)
                coeffs = pywt.wavedec(
                    upsampled, wavelet=wavelet, mode="periodization", level=level
                )
                return coeffs_to_flat(coeffs)

            def phi_pinv_op(measurements):
                upsampled = pseudo_inverse(measurements)
                coeffs = pywt.wavedec(
                    upsampled, wavelet=wavelet, mode="periodization", level=level
                )
                return coeffs_to_flat(coeffs)

            y = phi_op(sparse_signal)

            for _ in range(reconstruction_attempts):
                x_hat = classical_cosamp(
                    phi_op=phi_op,
                    phi_transpose_op=phi_transpose_op,
                    phi_pinv_op=phi_pinv_op,
                    y=y,
                    target_s=s,
                    n_coeffs=n_coeffs,
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
                if example_pair is None and m == plot_example_m:
                    example_pair = (sparse_signal.copy(), x_hat.copy(), m)

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
    filename = f"figures/sparse_subsampling_reconstruction_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    if show_plot:
        plt.show()
    if example_pair is not None:
        original, reconstructed, m_used = example_pair
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(original, label="Original coeffs", linewidth=1.2)
        ax.plot(reconstructed, label="Reconstructed coeffs", linewidth=1.2, alpha=0.8)
        ax.set_title(
            f"Coefficient-domain reconstruction example (m={m_used})", fontsize=12
        )
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if show_plot:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Subsampling Reconstruction (CoSaMP)")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display (still saves the figure).",
    )
    args = parser.parse_args()

    print("Subsampling Reconstruction (CoSaMP)")
    m_values = np.linspace(20, 500, 20).astype(int)

    generate_sparse_signal_reconstruction_data(
        n=1024,
        s=10,
        signal_count=10,
        reconstruction_attempts=2,
        m_values=m_values,
        show_plot=not args.no_show,
        plot_example_m=500,
    )


if __name__ == "__main__":
    main()
