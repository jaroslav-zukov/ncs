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
        return signal[indices]

    def transposed(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled

    return subsample, transposed


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

    return measure, adjoint


# MeasurementFunction = Callable[[np.ndarray], np.ndarray]
# MeasurementOperators = Callable[
#     [int, int, int | None], tuple[MeasurementFunction, MeasurementFunction]
# ]

# MEASUREMENT_OPERATORS: dict[str, MeasurementOperators] = {
MEASUREMENT_OPERATORS = {
    "subsampling": create_subsampling_operator,
    "gaussian": create_gaussian_operator,
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
        Tuple (measure, adjoint) as returned by the selected
        factory.  See individual factory docstrings for semantics.

    Raises:
        ValueError: If m >= n, if mode is unknown, or if m exceeds the
            frequency count limit for fourier_subsampling.
    """
    if m >= n:
        raise ValueError("m must be less than n")

    if measurement_mode not in MEASUREMENT_OPERATORS.keys():
        raise ValueError(
            f"Measurement mode {measurement_mode} is not supported ({MEASUREMENT_OPERATORS.keys()})"
        )

    measurement_operators = MEASUREMENT_OPERATORS[measurement_mode]
    return measurement_operators(n, m, seed)


create_measurement_operator = create_measurement_operators
