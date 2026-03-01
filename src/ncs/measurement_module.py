"""
Measurement operators for compressive sensing of nanopore signals.

This module provides three types of linear measurement operators Φ: ℝⁿ → ℝᵐ
(m << n) used to acquire compressed measurements y = Φx of a signal x.

Operator types
--------------
gaussian
    Dense i.i.d. Gaussian matrix with entries φᵢⱼ ~ N(0, 1/m).  Satisfies
    the Restricted Isometry Property (RIP) with high probability for any
    fixed sparsity basis, including wavelets.  Universal: RIP holds
    simultaneously for all sparse signals regardless of the basis used.
    References: Candès & Tao (2006), Baraniuk et al. (2008).

subsampling
    Selects m random time-domain samples from the length-n signal.
    **IMPORTANT — coherence limitation:** time-domain subsampling is
    *coherent* with the Haar/Daubechies wavelet basis.  Wavelet scaling
    functions are low-frequency and spread across many time samples, so a
    randomly chosen time sample "sees" contributions from many wavelet
    coefficients simultaneously.  There are **no known RIP guarantees**
    for the composition (time-domain subsampling) ∘ (wavelet synthesis).
    In practice recovery may still work for highly sparse signals, but
    the theoretical support is absent.

fourier_subsampling
    Selects m random Fourier (rfft) coefficients.  Fourier and wavelet
    bases are *incoherent*, so RIP guarantees exist for Fourier
    subsampling of wavelet-sparse signals (Candès et al., 2006;
    Lustig et al., 2007).  This is the theoretically sound alternative
    to time-domain subsampling for wavelet CS.

Normalisation conventions
--------------------------
All operators are normalised so that E[‖Φx‖²] ≈ ‖x‖² (energy-preserving
in expectation):
  - Gaussian: entries ~ N(0, 1/m), giving E[‖Φx‖²] = ‖x‖².
  - Subsampling: scaled by √(n/m), giving E[‖Φx‖²] = ‖x‖².
  - Fourier subsampling: rfft with norm="ortho" is unitary, so
    E[‖Φx‖²] = (m/n_freqs)·‖x‖² for random frequency selection.

Each factory returns a triple (measure, adjoint, pseudo_inverse):
  - measure(x)         → y = Φx
  - adjoint(y)         → Φᵀy (or Φ*y for complex operators)
  - pseudo_inverse(y)  → Φ†y = (ΦᵀΦ)⁻¹Φᵀy  (minimises ‖Φz - y‖)
"""

from typing import Callable

import numpy as np


def create_subsampling_operator(n: int, m: int, seed: int = None):
    """
    Create a time-domain random subsampling measurement operator.

    Randomly selects m out of n time-domain sample indices (without
    replacement) and rescales by √(n/m) so that the operator is
    energy-preserving in expectation: E[‖Φx‖²] = ‖x‖².

    **Coherence warning:** This operator is coherent with wavelet bases.
    Wavelet basis vectors (scaling functions / detail functions) are spread
    across many time samples, so each measurement implicitly mixes many
    wavelet coefficients.  The mutual coherence μ(Φ, Ψ) = √(n/m)·max|φᵢ·ψⱼ|
    can be O(1) for Haar wavelets, violating the incoherence requirement for
    RIP-based recovery guarantees.  Use fourier_subsampling for wavelet CS
    with theoretical backing.

    Args:
        n: Signal length (number of time-domain samples).
        m: Number of measurements to take (m < n).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of three callables (subsample, transposed, pseudo_inverse):

        subsample(signal) → np.ndarray of shape (m,)
            Forward operator: selects m samples and scales by √(n/m).
            Mathematically: y = S·x·√(n/m) where S ∈ {0,1}^{m×n} is the
            row-selection matrix for the chosen indices.

        transposed(measurements) → np.ndarray of shape (n,)
            Adjoint / transpose operator: scatters m measurements back into
            an n-dimensional vector at the selected indices and scales by
            √(n/m).  This is Sᵀ·y·√(n/m).  Note: transposed ∘ subsample ≠ I.

        pseudo_inverse(measurements) → np.ndarray of shape (n,)
            Pseudo-inverse: same as transposed but scales by √(m/n) instead
            of √(n/m).  Equivalently Sᵀ·y·√(m/n) = S†·y where S† = Sᵀ/(n/m)
            is the Moore–Penrose pseudo-inverse of the normalised operator.
            Satisfies subsample(pseudo_inverse(y)) = y.
    """
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))

    def subsample(signal: np.ndarray) -> np.ndarray:
        return signal[indices] * np.sqrt(n / m)

    def transposed(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled * np.sqrt(n / m)

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled * np.sqrt(m / n)

    return subsample, transposed, pseudo_inverse


def create_gaussian_operator(n: int, m: int, seed: int = None):
    """
    Create a dense i.i.d. Gaussian measurement operator.

    Constructs an m×n matrix Φ with entries drawn i.i.d. from N(0, 1/m).
    The 1/√m normalisation ensures E[‖Φx‖²] = ‖x‖² for any fixed x.

    Gaussian matrices satisfy the Restricted Isometry Property (RIP) of
    order k with high probability when m = O(k log(n/k)), *independently*
    of the sparsity basis Ψ.  This universality makes Gaussian operators
    suitable for wavelet-domain CS without any coherence concerns.
    References: Baraniuk et al. (2008), Candès & Tao (2006).

    The pseudo-inverse Φ† = (ΦᵀΦ)⁻¹Φᵀ is computed once via
    np.linalg.pinv and reused across calls (O(mn²) precomputation).

    Args:
        n: Signal / coefficient vector length.
        m: Number of measurements (m < n).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of three callables (measure, adjoint, pseudo_inverse):

        measure(signal) → np.ndarray of shape (m,)
            y = Φ·x, standard matrix–vector product.

        adjoint(measurements) → np.ndarray of shape (n,)
            Φᵀ·y, transpose matrix–vector product.

        pseudo_inverse(measurements) → np.ndarray of shape (n,)
            Φ†·y = (ΦᵀΦ)⁻¹Φᵀ·y, least-squares solution to Φz = y.
    """
    rng = np.random.default_rng(seed)
    phi = rng.normal(0, 1.0 / np.sqrt(m), size=(m, n))
    phi_pinv = np.linalg.pinv(phi)

    def measure(signal: np.ndarray) -> np.ndarray:
        return phi @ signal

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        return phi.T @ measurements

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        return phi_pinv @ measurements

    return measure, adjoint, pseudo_inverse


def create_fourier_subsampling_operator(n: int, m: int, seed: int = None):
    """
    Creates a Fourier subsampling measurement operator.

    Samples m random Fourier coefficients from the real-valued signal of length n.
    Uses rfft (only unique frequencies: n//2+1 total), so m must be <= n//2+1.

    This operator is incoherent with the wavelet basis, providing RIP guarantees
    for signals sparse in the wavelet domain (Candès et al., 2006).
    Contrast with time-domain subsampling which is coherent with wavelets and
    does NOT have known RIP guarantees.

    The measurement model is y = S · DFT · x, where S selects m random frequency
    indices. Since x is real-valued, DFT output is Hermitian symmetric: only the
    first n//2+1 unique frequencies are needed (rfft). With norm="ortho", rfft is
    a unitary transform, so the adjoint equals the pseudo-inverse.

    References:
        Candès, E., Romberg, J., & Tao, T. (2006). Robust uncertainty principles.
        Lustig, M., Donoho, D., & Pauly, J. (2007). Sparse MRI.

    Args:
        n: Signal length (should be power of 2 for wavelet compatibility)
        m: Number of Fourier measurements (must be <= n//2+1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (measure, adjoint, pseudo_inverse) callables

    Raises:
        ValueError: If m > n//2+1 (more measurements than unique frequencies)
    """
    rng = np.random.default_rng(seed)
    n_freqs = n // 2 + 1  # number of unique rfft frequencies
    if m > n_freqs:
        raise ValueError(f"m={m} exceeds number of unique frequencies n//2+1={n_freqs}")

    freq_indices = np.sort(rng.choice(n_freqs, size=m, replace=False))

    def measure(signal: np.ndarray) -> np.ndarray:
        """Apply S · DFT: compute rfft then select m frequencies."""
        full_fft = np.fft.rfft(signal, norm="ortho")
        return full_fft[freq_indices]

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        """Apply (S · rfft)^T: correct real adjoint of the subsampled rfft.

        rfft(x)[k] == fft(x)[k] for k = 0..n//2.  The real adjoint of the map
        x → fft(x, ortho)[freq_indices] satisfies:
            Re(<Phi x, y>_C) = <x, Phi^T y>_R
        and is computed as:
            y → Re(ifft(Y, ortho))
        where Y[freq_indices[k]] = y[k] and Y is zero elsewhere.
        Note: we do NOT use irfft, which would incorrectly double-count interior
        frequency contributions via implicit Hermitian symmetry extension.
        """
        full_fft = np.zeros(n, dtype=complex)
        full_fft[freq_indices] = measurements
        return np.fft.ifft(full_fft, n=n, norm="ortho").real

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        """
        Pseudo-inverse of S · rfft.
        Since rfft is unitary (ortho norm) and S is a row-selection matrix,
        the pseudo-inverse of (S · rfft) equals the adjoint.
        """
        return adjoint(measurements)

    return measure, adjoint, pseudo_inverse


# MeasurementFunction = Callable[[np.ndarray], np.ndarray]
# MeasurementOperators = Callable[
#     [int, int, int | None], tuple[MeasurementFunction, MeasurementFunction]
# ]

# MEASUREMENT_OPERATORS: dict[str, MeasurementOperators] = {
MEASUREMENT_OPERATORS = {
    "subsampling": create_subsampling_operator,
    "gaussian": create_gaussian_operator,
    "fourier_subsampling": create_fourier_subsampling_operator,
}


def create_measurement_operators(
    measurement_mode: str,
    n: int,
    m: int,
    seed: int | None = None,
):
    """
    Dispatcher: construct a measurement operator triple by name.

    Validates parameters and delegates to the appropriate factory function
    (create_subsampling_operator, create_gaussian_operator, or
    create_fourier_subsampling_operator).

    Args:
        measurement_mode: One of "subsampling", "gaussian",
            "fourier_subsampling".  See module docstring for RIP properties
            and coherence considerations of each mode.
        n: Signal length.
        m: Number of measurements; must satisfy m < n (and m ≤ n//2+1 for
            fourier_subsampling).
        seed: Optional random seed forwarded to the chosen factory.

    Returns:
        Tuple (measure, adjoint, pseudo_inverse) as returned by the selected
        factory.  See individual factory docstrings for semantics.

    Raises:
        ValueError: If m >= n, if mode is unknown, or if m exceeds the
            frequency count limit for fourier_subsampling.
    """
    if m >= n:
        raise ValueError("m must be less than n")

    if measurement_mode == "fourier_subsampling":
        n_freqs = n // 2 + 1
        if m > n_freqs:
            raise ValueError(
                f"m={m} exceeds number of unique frequencies n//2+1={n_freqs} "
                "for fourier_subsampling mode"
            )

    if measurement_mode not in MEASUREMENT_OPERATORS.keys():
        raise ValueError(
            f"Measurement mode {measurement_mode} is not supported ({MEASUREMENT_OPERATORS.keys()})"
        )

    measurement_operators = MEASUREMENT_OPERATORS[measurement_mode]
    return measurement_operators(n, m, seed)
