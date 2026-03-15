"""
Prototype: CoSaMP reconstruction from time-domain subsampling with wavelet sparsity.

This self-contained script investigates how well classical CoSaMP (without
tree-sparsity structure) can recover a *wavelet-sparse* signal from
*time-domain subsampling* measurements.

What it does
------------
1. Generates a synthetic signal that is k-sparse in the wavelet coefficient
   domain (random non-zero coefficients at random positions).
2. Applies a time-domain subsampling operator (select m out of n samples,
   scaled by √(n/m)) to obtain compressed measurements y.
3. Runs classical CoSaMP (standard ℓ₀-sparse projection, no tree model) in
   the wavelet coefficient domain by composing the operators with the DWT/IDWT.
4. Evaluates reconstruction quality (relative ℓ₂ error, missed support) over
   a range of measurement counts m and across multiple random signals.
5. Produces a scatter+mean-line plot of relative error vs m, coloured by
   missed support count, and saves it to figures/.

RIP / coherence limitation
--------------------------
**This prototype is expected to fail or degrade** for a fundamental reason:
time-domain subsampling and the wavelet basis are *coherent*.  Wavelet basis
vectors (scaling and detail functions) have significant energy at many time
samples simultaneously, so random time-domain measurements do NOT satisfy the
Restricted Isometry Property (RIP) for wavelet-sparse signals.

Without RIP, CoSaMP's convergence guarantee does not hold.  In practice the
algorithm may recover very sparse signals by luck (when the support happens to
align with low-coherence time samples), but performance degrades rapidly as
sparsity or measurement count changes.

The correct approach for wavelet-sparse CS is Fourier subsampling (incoherent
with wavelets; see measurement_module.create_fourier_subsampling_operator) or
Gaussian measurements.  This prototype serves as an empirical baseline /
sanity check demonstrating the coherence failure.

Output plots
------------
figures/sparse_subsampling_reconstruction_<timestamp>.png
    Scatter plot (individual trials) + mean relative error vs m.
    Colour encodes missed support size (0 = perfect support recovery).
    A second figure shows an example original vs reconstructed coefficient
    vector for m = plot_example_m.
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

from ncs.measurement_module import create_subsampling_operator


def generate_sparse_x(n, s):
    """
    Generate a random s-sparse signal of length n in the wavelet coefficient domain.

    Args:
        n: Total signal / coefficient length.
        s: Number of non-zero coefficients (sparsity level).

    Returns:
        np.ndarray: Length-n array with s random non-zero entries drawn
            uniformly from integers in [-300, 300] at random positions.
    """
    signal = np.zeros(n)
    indices = np.sort(np.random.choice(n, s, replace=False))
    # signal[indices] = np.random.random(s) # float from 0 to 1
    signal[indices] = np.random.randint(-300, 300, size=s) # integer from -300 to 300
    return signal


def sparse_projection(array, sparsity):
    """
    Hard-threshold an array to keep only the `sparsity` largest-magnitude entries.

    Args:
        array: Input numpy array.
        sparsity: Number of non-zero entries to retain.

    Returns:
        np.ndarray: Copy of array with all but the top-`sparsity` absolute
            values set to zero.
    """
    result = np.zeros_like(array)
    indices = np.argsort(np.abs(array))[-sparsity:]
    result[indices] = array[indices]
    return result


def support(array):
    """
    Return the set of indices where array is non-zero.

    Args:
        array: 1-D numpy array.

    Returns:
        set[int]: 0-based indices with non-zero values.
    """
    return set(np.nonzero(array)[0])


@lru_cache(maxsize=None)
def coeff_array_spec(signal_length, wavelet):
    """
    Compute and cache wavelet coefficient array metadata for a given signal length.

    Used to convert between flat coefficient arrays and the grouped pywt
    coefficient list format.  The result is cached to avoid repeated pywt
    calls during the inner reconstruction loop.

    Args:
        signal_length: Length of the time-domain signal (should be power of 2).
        wavelet: Wavelet name string (e.g. 'haar').

    Returns:
        Tuple (shape, coeff_slices, max_level):
            shape: Shape tuple of the 1-D flattened coefficient array.
            coeff_slices: pywt coefficient slice list for array_to_coeffs.
            max_level: Maximum DWT decomposition level used.
    """
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
    """
    Convert a flat coefficient array to pywt grouped coefficient list.

    Args:
        flat_coeffs: 1-D array of wavelet coefficients (length = prod(shape)).
        shape: Expected shape tuple from coeff_array_spec.
        coeff_slices: Slice list from coeff_array_spec for pywt.array_to_coeffs.

    Returns:
        list[np.ndarray]: pywt.wavedec-compatible coefficient list.

    Raises:
        ValueError: If flat_coeffs length does not match shape.
    """
    if np.prod(shape) != len(flat_coeffs):
        raise ValueError(
            f"Flat coeffs length {len(flat_coeffs)} does not match wavelet shape {shape}."
        )
    coeff_arr = np.asarray(flat_coeffs).reshape(shape)
    return pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format="wavedec")


def coeffs_to_flat(coeffs):
    """
    Convert a pywt grouped coefficient list to a flat 1-D numpy array.

    Args:
        coeffs: pywt.wavedec-compatible coefficient list (list of arrays).

    Returns:
        np.ndarray: 1-D ravelled coefficient array.
    """
    coeff_arr, _ = pywt.coeffs_to_array(coeffs)
    return coeff_arr.ravel()


def project_on_support(array, support_indices):
    """
    Zero out all entries of array except those at support_indices.

    Args:
        array: Input 1-D numpy array.
        support_indices: Array-like of integer indices to retain.

    Returns:
        np.ndarray: Copy of array with all entries outside support_indices
            set to zero.
    """
    projected = np.zeros_like(array)
    projected[support_indices] = array[support_indices]
    return projected


def classical_cosamp(phi_op, phi_transpose_op, phi_pinv_op, y, target_s, n_coeffs):
    """
    Classical CoSaMP algorithm with standard ℓ₀ (flat) sparsity projection.

    Implements CoSaMP (Needell & Tropp, 2009) without any structured sparsity
    model (no tree projection).  The 2k-sparse proxy identification and k-sparse
    pruning both use simple hard-thresholding (keep top magnitudes).

    Algorithm per iteration (up to 50 iterations):
    1. **Proxy** e = Φᵀr
       Apply the adjoint operator to the current residual r = y − Φx̂.
    2. **Identify 2k support** Ω = support(hard_threshold(e, 2·target_s))
       Select the 2·target_s largest-magnitude positions of e.
    3. **Merge** T = support(x̂) ∪ Ω
       Union of current estimate support and new candidates.
    4. **Least squares on T** b = project_on_support(Φ†y, T)
       Apply pseudo-inverse to measurements, then restrict to T.
    5. **Prune** x̂ = hard_threshold(b, target_s)
       Keep only target_s largest-magnitude entries.
    6. **Update residual** r = y − Φx̂
       If ‖r‖ < 1e-10, declare convergence and stop early.

    Args:
        phi_op: Callable, forward measurement operator x_flat → y.
        phi_transpose_op: Callable, adjoint operator y → x_flat.
        phi_pinv_op: Callable, pseudo-inverse operator y → x_flat.
        y: Measurement vector (1-D numpy array, length m).
        target_s: Target sparsity level k.
        n_coeffs: Length of the coefficient / signal vector.

    Returns:
        np.ndarray: Estimated sparse coefficient vector x̂ of length n_coeffs.
    """
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
    """
    Run CoSaMP reconstruction experiments and produce a summary plot.

    For each value of m in m_values, generates signal_count random s-sparse
    wavelet-domain signals, acquires m time-domain subsampling measurements,
    and runs CoSaMP reconstruction_attempts times each.  Records relative ℓ₂
    error and missed support count.

    **Note on expected results:** Due to the coherence of time-domain
    subsampling with the Haar wavelet basis (see module docstring), recovery
    quality is expected to be poor or unreliable compared to Gaussian or
    Fourier subsampling measurements.

    Args:
        n: Signal length (should be power of 2, e.g. 1024).
        s: Sparsity level of the ground-truth wavelet coefficient vector.
        signal_count: Number of random signals to generate per m value.
        reconstruction_attempts: Number of CoSaMP runs per (signal, m) pair.
            Since CoSaMP is deterministic for fixed inputs, this parameter
            has no effect on diversity but may be used for future stochastic
            variants.
        m_values: 1-D array of measurement counts to sweep over (e.g.
            np.linspace(20, 500, 20).astype(int)).
        show_plot: If True, display the plot interactively (plt.show()).
            Set to False in headless / CI environments (--no-show flag).
        plot_example_m: The specific m value for which to show an example
            original vs. reconstructed coefficient plot.  Must be present
            in m_values (or the example plot will be skipped).

    Output
    ------
    Saves two plots to figures/:
      1. ``sparse_subsampling_reconstruction_<YYYYMMDD_HHMM>.png``
         Scatter plot of relative error vs m (individual trials coloured by
         missed support count) with a blue mean-error trend line.
      2. Coefficient-domain reconstruction example for m = plot_example_m
         (original vs reconstructed flat wavelet coefficients).

    Returns:
        None (results are plotted and saved to disk).
    """
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
