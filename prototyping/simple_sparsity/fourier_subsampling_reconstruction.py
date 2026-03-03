"""
Prototype: Fourier subsampling reconstruction of wavelet-sparse signals via CoSaMP.

This is a **self-contained prototype** demonstrating that Fourier subsampling
is incoherent with the wavelet basis and achieves reliable recovery where
time-domain subsampling fails.

Motivation
----------
Time-domain subsampling and wavelet sparsity are *coherent*: wavelet basis
functions have compact time support, so a random subsampling pattern can miss
a wavelet's support region entirely, yielding zero information about that
coefficient.  This causes RIP to fail and CoSaMP to break down.

The fix — Fourier subsampling — comes from the classic MRI compressed sensing
result (Candès, Romberg & Tao 2006; Lustig, Donoho & Pauly 2007):

    y = S · DFT · x

where S selects m random Fourier coefficients.  DFT basis functions are
globally spread across time, making them maximally incoherent with
time-localised wavelets.  This gives provable RIP guarantees with
m = O(s log n) measurements.

This script sweeps over measurement counts m and compares:
- Gaussian measurements  (universal RIP, expensive O(mn) matrix-vector products)
- Fourier subsampling    (incoherent with wavelets, cheap O(n log n) FFT)

Usage
-----
Run directly::

    python fourier_subsampling_reconstruction.py [--no-show]

The script saves a comparison plot to ``figures/``.

References
----------
- Candès, E., Romberg, J., & Tao, T. (2006). Robust uncertainty principles:
  Exact signal reconstruction from highly incomplete frequency information.
  IEEE Transactions on Information Theory, 52(2), 489–509.
- Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application
  of compressed sensing for rapid MR imaging. Magnetic Resonance in Medicine,
  58(6), 1182–1195.
- Tropp, J. A., et al. (2010). Beyond Nyquist: Efficient sampling of sparse
  bandlimited signals. IEEE Transactions on Information Theory, 56(1), 520–544.
"""

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

from ncs.measurement_module import (
    create_fourier_subsampling_operator,
    create_gaussian_operator,
)


def generate_sparse_x(n, s, rng=None):
    """
    Generate a random s-sparse signal of length n in the wavelet coefficient domain.

    Non-zero entries are drawn uniformly from the integers in [-300, 300].
    """
    if rng is None:
        rng = np.random.default_rng()
    signal = np.zeros(n)
    indices = np.sort(rng.choice(n, s, replace=False))
    signal[indices] = rng.integers(-300, 300, size=s)
    return signal


def sparse_projection(array, sparsity):
    """Hard-threshold ``array`` to its ``sparsity`` largest-magnitude entries."""
    result = np.zeros_like(array)
    # Handle complex arrays (Fourier measurements produce complex residuals
    # in phi_transpose, but since we work in wavelet coeff domain the result
    # should be real; take real part defensively).
    mag = np.abs(np.real(array))
    indices = np.argsort(mag)[-sparsity:]
    result[indices] = array[indices]
    return np.real(result)


def support(array):
    """Return the set of indices where ``array`` is non-zero."""
    return set(np.nonzero(np.real(array))[0])


def project_on_support(array, support_indices):
    """Zero out all entries of ``array`` outside ``support_indices``."""
    projected = np.zeros_like(array)
    if len(support_indices) > 0:
        projected[support_indices] = array[support_indices]
    return projected


@lru_cache(maxsize=None)
def coeff_array_spec(signal_length, wavelet):
    """Compute and cache the wavelet coefficient array shape and slices."""
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
    """Convert a flat coefficient array to PyWavelets coefficient list format."""
    coeff_arr = np.asarray(flat_coeffs).reshape(shape)
    return pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedec")


def coeffs_to_flat(coeffs):
    """Flatten a PyWavelets coefficient list to a 1-D array."""
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    return coeff_arr.ravel()


def classical_cosamp(phi_op, phi_transpose_op, phi_pinv_op, y, target_s, n_coeffs):
    """
    Plain (non-tree-sparse) CoSaMP for flat wavelet coefficient vectors.

    Implements Needell & Tropp (2009) CoSaMP with hard-thresholding.
    Works entirely in the wavelet coefficient domain via composed operators.

    Parameters
    ----------
    phi_op : callable
        Forward operator in coefficient space: x_flat ↦ measure(IDWT(x_flat)).
        For Fourier subsampling, output is complex.
    phi_transpose_op : callable
        Adjoint in coefficient space: y ↦ DWT(adjoint_measure(y)).
    phi_pinv_op : callable
        Pseudo-inverse in coefficient space: y ↦ DWT(pinv_measure(y)).
        For Fourier subsampling (unitary DFT), equals phi_transpose_op.
    y : np.ndarray
        Measurement vector (complex for Fourier subsampling).
    target_s : int
        Target sparsity.
    n_coeffs : int
        Total number of wavelet coefficients.
    """
    x_hat = np.zeros(n_coeffs)
    r = y.copy()

    for _ in range(50):
        e = phi_transpose_op(r)

        omega_e = np.nonzero(sparse_projection(np.real(e), 2 * target_s))[0]

        current_support = np.nonzero(x_hat)[0]
        t = np.union1d(current_support, omega_e).astype(int)

        b = project_on_support(np.real(phi_pinv_op(y)), t)

        x_hat = sparse_projection(array=b, sparsity=target_s)

        r = y - phi_op(x_hat)

        if np.linalg.norm(r) < 1e-10:
            break

    return x_hat


def run_fourier_experiment(n, s, signal_count, reconstruction_attempts, m_values, wavelet="haar"):
    """Run Fourier subsampling CoSaMP experiment, return DataFrame of results."""
    results = []
    shape, coeff_slices, level = coeff_array_spec(n, wavelet)
    n_coeffs = int(np.prod(shape))
    rng = np.random.default_rng(42)

    for m in tqdm(m_values, desc="Fourier subsampling"):
        for _ in range(signal_count):
            measure, adjoint, pseudo_inv = create_fourier_subsampling_operator(n, m, seed=int(rng.integers(1e9)))
            sparse_signal = generate_sparse_x(n_coeffs, s, rng)

            def phi_op(x_flat):
                coeffs = flat_to_coeffs(x_flat, shape, coeff_slices)
                signal = pywt.waverec(coeffs, wavelet=wavelet, mode="periodization")
                return measure(signal)

            def phi_transpose_op(measurements):
                upsampled = adjoint(measurements)
                coeffs = pywt.wavedec(upsampled, wavelet=wavelet, mode="periodization", level=level)
                return coeffs_to_flat(coeffs)

            def phi_pinv_op(measurements):
                upsampled = pseudo_inv(measurements)
                coeffs = pywt.wavedec(upsampled, wavelet=wavelet, mode="periodization", level=level)
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
                norm_signal = np.linalg.norm(sparse_signal)
                relative_error = np.linalg.norm(sparse_signal - x_hat) / norm_signal
                missed = len(support(sparse_signal) - support(x_hat))
                results.append({"m": m, "relative_error": relative_error, "missed_support": missed, "method": "Fourier subsampling"})

    return pd.DataFrame(results)


def run_gaussian_experiment(n, s, signal_count, reconstruction_attempts, m_values, wavelet="haar"):
    """Run Gaussian measurement CoSaMP experiment, return DataFrame of results."""
    results = []
    shape, coeff_slices, level = coeff_array_spec(n, wavelet)
    n_coeffs = int(np.prod(shape))
    rng = np.random.default_rng(123)

    for m in tqdm(m_values, desc="Gaussian measurements"):
        for _ in range(signal_count):
            measure, adjoint, pseudo_inv = create_gaussian_operator(n_coeffs, m, seed=int(rng.integers(1e9)))
            sparse_signal = generate_sparse_x(n_coeffs, s, rng)

            y = measure(sparse_signal)

            for _ in range(reconstruction_attempts):
                x_hat = classical_cosamp(
                    phi_op=measure,
                    phi_transpose_op=adjoint,
                    phi_pinv_op=pseudo_inv,
                    y=y,
                    target_s=s,
                    n_coeffs=n_coeffs,
                )
                norm_signal = np.linalg.norm(sparse_signal)
                relative_error = np.linalg.norm(sparse_signal - x_hat) / norm_signal
                missed = len(support(sparse_signal) - support(x_hat))
                results.append({"m": m, "relative_error": relative_error, "missed_support": missed, "method": "Gaussian"})

    return pd.DataFrame(results)


def generate_comparison_plot(n, s, signal_count, reconstruction_attempts, m_values, show_plot=True):
    """
    Sweep over m and compare Fourier subsampling vs Gaussian measurement recovery.

    Saves a side-by-side comparison plot to ``figures/``.
    """
    df_fourier = run_fourier_experiment(n, s, signal_count, reconstruction_attempts, m_values)
    df_gaussian = run_gaussian_experiment(n, s, signal_count, reconstruction_attempts, m_values)
    df = pd.concat([df_fourier, df_gaussian], ignore_index=True)

    params = {
        "N": n,
        "Sparsity (s)": s,
        "Signals per m": signal_count,
        "Attempts per signal": reconstruction_attempts,
    }
    param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, method in zip(axes, ["Fourier subsampling", "Gaussian"]):
        sub_df = df[df["method"] == method]
        sns.stripplot(
            data=sub_df, x="m", y="relative_error", hue="missed_support",
            jitter=0.2, alpha=0.6, palette="RdYlGn_r", native_scale=True, ax=ax,
        )
        sns.lineplot(
            data=sub_df, x="m", y="relative_error", estimator="mean",
            color="blue", linewidth=2, errorbar=None, ax=ax, label="Mean error",
        )
        ax.set_title(method, fontsize=13)
        ax.set_xlabel("Number of measurements m", fontsize=11)
        ax.set_ylabel("Relative error ‖x − x̂‖ / ‖x‖" if ax is axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Missed support", fontsize=9)

    at = AnchoredText(s=param_str, loc="upper right", bbox_to_anchor=(1, 0.72),
                      bbox_transform=axes[1].transAxes, frameon=True)
    axes[1].add_artist(at)

    fig.suptitle(
        "Compressed Sensing Recovery: Fourier Subsampling vs Gaussian\n"
        "(wavelet-sparse signals, CoSaMP, Haar wavelet)",
        fontsize=14,
    )
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"figures/fourier_vs_gaussian_reconstruction_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")
    if show_plot:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Fourier Subsampling Reconstruction (CoSaMP)")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive plot display (still saves the figure).",
    )
    args = parser.parse_args()

    print("Fourier Subsampling vs Gaussian Reconstruction (CoSaMP)")
    n = 1024
    # Maximum Fourier measurements = n//2+1 = 513; keep m_values well below this
    m_values = np.linspace(20, 400, 15).astype(int)

    generate_comparison_plot(
        n=n,
        s=10,
        signal_count=5,
        reconstruction_attempts=2,
        m_values=m_values,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
