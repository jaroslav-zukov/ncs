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
import pywt


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _sylvester_hadamard(n: int) -> np.ndarray:
    matrix = np.array([[1.0]])
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _sequency_ordered_hadamard(n: int) -> np.ndarray:
    if not _is_power_of_two(n):
        raise ValueError("Hadamard operators require n to be a power of 2")

    try:
        from scipy.linalg import hadamard  # type: ignore

        hadamard_matrix = hadamard(n, dtype=float)
    except Exception:
        hadamard_matrix = _sylvester_hadamard(n)

    sign_change_counts = np.sum(
        hadamard_matrix[:, 1:] != hadamard_matrix[:, :-1], axis=1
    )
    sequency_order = np.argsort(sign_change_counts, kind="stable")
    return hadamard_matrix[sequency_order]


def _wavelet_alpha_nu(wavelet: str) -> tuple[float, float]:
    wavelet_obj = pywt.Wavelet(wavelet)
    alpha_overrides = {
        "coif9": 3.2,
    }
    alpha = alpha_overrides.get(wavelet, max((wavelet_obj.dec_len - 1) / 2.0, 1.0))
    nu = float(wavelet_obj.vanishing_moments_psi or 1.0)
    return float(alpha), nu


def _largest_remainder_capped(
    raw: np.ndarray,
    total: int,
    capacities: np.ndarray,
) -> np.ndarray:
    allocation = np.floor(raw).astype(int)
    allocation = np.minimum(allocation, capacities)

    deficit = int(total - np.sum(allocation))
    if deficit > 0:
        fractional = raw - np.floor(raw)
        for idx in np.argsort(-fractional):
            available = int(capacities[idx] - allocation[idx])
            if available <= 0:
                continue
            take = min(available, deficit)
            allocation[idx] += take
            deficit -= take
            if deficit == 0:
                break

    if deficit > 0:
        for idx in np.argsort(raw)[::-1]:
            available = int(capacities[idx] - allocation[idx])
            if available <= 0:
                continue
            take = min(available, deficit)
            allocation[idx] += take
            deficit -= take
            if deficit == 0:
                break

    if int(np.sum(allocation)) != total:
        raise ValueError("Internal allocation error: sampled row count does not equal m")

    return allocation


def _allocate_multilevel_samples(
    band_sizes: np.ndarray,
    m: int,
    wavelet: str,
    local_sparsities: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r = len(band_sizes)
    if m < 1:
        raise ValueError("m must be positive")
    if m > int(np.sum(band_sizes)):
        raise ValueError("m exceeds available sequency coefficients in dyadic bands")

    alpha, nu = _wavelet_alpha_nu(wavelet)

    if local_sparsities is not None:
        sparsities = np.asarray(local_sparsities, dtype=float)
        if sparsities.shape != (r,):
            raise ValueError(f"local_sparsities must have length {r}")
        if np.any(sparsities < 0):
            raise ValueError("local_sparsities must be non-negative")
    else:
        sparsities = np.ones(r, dtype=float)

    n_ends = np.cumsum(band_sizes).astype(float)
    n_prev = np.concatenate(([1.0], n_ends[:-1]))
    prefactor = (n_ends - n_prev) / n_prev

    weights = np.zeros(r, dtype=float)
    for level_k in range(r):
        left = max(0, level_k - 1)
        right = min(r - 1, level_k + 1)
        neighbor_max = float(np.max(sparsities[left : right + 1]))

        coarse_to_fine = 0.0
        for level_l in range(level_k):
            a_kl = level_k - level_l
            coarse_to_fine += float(sparsities[level_l] * 2.0 ** (-(alpha - 0.5) * a_kl))

        fine_to_coarse = 0.0
        for level_l in range(level_k + 1, r):
            b_kl = level_l - level_k
            fine_to_coarse += float(sparsities[level_l] * 2.0 ** (-nu * b_kl))

        theorem_term = neighbor_max + coarse_to_fine + fine_to_coarse
        weights[level_k] = max(float(prefactor[level_k] * theorem_term), 0.0)

    if np.all(weights == 0):
        weights = np.ones(r, dtype=float)

    normalized_weights = weights / np.sum(weights)
    raw = m * normalized_weights
    allocation = _largest_remainder_capped(raw=raw, total=m, capacities=band_sizes)

    return allocation, normalized_weights, alpha, nu


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
    scale = np.sqrt(n / m)

    def subsample(signal: np.ndarray) -> np.ndarray:
        return signal[indices] * scale

    def transposed(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled * scale

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled / scale

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
    def measure(signal: np.ndarray) -> np.ndarray:
        return phi @ signal

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        return phi.T @ measurements

    phi_pinv = np.linalg.pinv(phi)

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        return phi_pinv @ measurements

    return measure, adjoint, pseudo_inverse


def _create_gaussian_operator_with_pinv(n: int, m: int, seed: int = None):
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


def create_hadamard_operator(n: int, m: int, seed: int = None):
    """
    Create a uniform subsampled Walsh-Hadamard measurement operator.

    The transform uses Walsh sequency ordering (rows sorted by sign changes),
    not natural Hadamard ordering. This preserves the low/high-sequency
    interpretation required for multilevel sampling strategies.

    Args:
        n: Signal length (must be a power of 2).
        m: Number of Hadamard coefficients to sample uniformly at random.
        seed: Optional random seed.

    Returns:
        Tuple of (measure, adjoint, pseudo_inverse).

    Raises:
        ValueError: If n is not power-of-two or m is out of range.
    """
    if not _is_power_of_two(n):
        raise ValueError("Hadamard operator requires n to be a power of 2")
    if m < 1 or m > n:
        raise ValueError("m must satisfy 1 <= m <= n")

    rng = np.random.default_rng(seed)
    hadamard_seq = _sequency_ordered_hadamard(n) / np.sqrt(n)
    indices = np.sort(rng.choice(n, size=m, replace=False))
    scale = np.sqrt(n / m)

    def measure(signal: np.ndarray) -> np.ndarray:
        transformed = hadamard_seq @ signal
        return transformed[indices] * scale

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        coeffs = np.zeros(n)
        coeffs[indices] = measurements * scale
        return hadamard_seq.T @ coeffs

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        coeffs = np.zeros(n)
        coeffs[indices] = measurements / scale
        return hadamard_seq.T @ coeffs

    return measure, adjoint, pseudo_inverse


def create_hadamard_multilevel_operator(
    n: int,
    m: int,
    wavelet: str = "coif9",
    local_sparsities: np.ndarray | None = None,
    seed: int = None,
):
    """
    Create a multilevel subsampled Walsh-Hadamard measurement operator.

    Sequency coefficients are partitioned into r = log2(n) dyadic bands
    that cover all rows:
        [(0, 2), (2, 4), (4, 8), ..., (n/2, n)).
    Within each band, rows are sampled uniformly at random using per-band
    allocations m_k.

    If local_sparsities is provided, allocation follows a Theorem 6.2-style
    wavelet-regularity model (Adcock et al.) using smoothness alpha and
    vanishing moments nu. Otherwise a neutral local sparsity profile is used.

    Args:
        n: Signal length (must be power-of-two).
        m: Total number of measurements.
        wavelet: Wavelet name used for regularity parameters (default: coif9).
        local_sparsities: Optional array of length r.
        seed: Optional random seed.

    Returns:
        Tuple (measure, adjoint, pseudo_inverse, metadata) where metadata
        contains per-band allocations, normalized allocation weights, wavelet
        regularity constants, and band boundaries.
    """
    if not _is_power_of_two(n):
        raise ValueError("Hadamard multilevel operator requires n power-of-two")
    if m < 1 or m >= n:
        raise ValueError("m must satisfy 1 <= m < n")

    rng = np.random.default_rng(seed)
    hadamard_seq = _sequency_ordered_hadamard(n) / np.sqrt(n)

    r = int(np.log2(n))
    band_boundaries: list[tuple[int, int]] = [(0, 2)]
    for level in range(2, r + 1):
        band_boundaries.append((2 ** (level - 1), 2**level))
    band_sizes = np.array([end - start for start, end in band_boundaries], dtype=int)

    allocations, allocation_weights, alpha, nu = _allocate_multilevel_samples(
        band_sizes, m, wavelet, local_sparsities
    )

    selected_indices: list[np.ndarray] = []
    for (start, end), band_m in zip(band_boundaries, allocations):
        if band_m <= 0:
            continue
        selected = np.sort(rng.choice(np.arange(start, end), size=band_m, replace=False))
        selected_indices.append(selected)

    if selected_indices:
        indices = np.sort(np.concatenate(selected_indices))
    else:
        indices = np.array([], dtype=int)

    scale = np.sqrt(n / m)

    def measure(signal: np.ndarray) -> np.ndarray:
        transformed = hadamard_seq @ signal
        return transformed[indices] * scale

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        coeffs = np.zeros(n)
        coeffs[indices] = measurements * scale
        return hadamard_seq.T @ coeffs

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        coeffs = np.zeros(n)
        coeffs[indices] = measurements / scale
        return hadamard_seq.T @ coeffs

    metadata = {
        "wavelet": wavelet,
        "band_boundaries": band_boundaries,
        "allocation": allocations,
        "allocation_weights": allocation_weights,
        "alpha": alpha,
        "nu": nu,
        "indices": indices,
    }
    return measure, adjoint, pseudo_inverse, metadata


# MeasurementFunction = Callable[[np.ndarray], np.ndarray]
# MeasurementOperators = Callable[
#     [int, int, int | None], tuple[MeasurementFunction, MeasurementFunction]
# ]

# MEASUREMENT_OPERATORS: dict[str, MeasurementOperators] = {
def create_random_modulation_operator(n: int, m: int, seed: int = None):
    """
    Creates a random modulation + subsampling measurement operator.

    Implements Option B / the "random demodulator" (Tropp et al., 2010):
        y = S · (r ⊙ x)
    where r ∈ {±1}^n is a random Rademacher chipping sequence and S
    selects m random time-domain indices.

    WHY THIS WORKS: Plain time-domain subsampling is coherent with the wavelet
    basis (wavelets are time-localized → subsampling can miss entire wavelet
    supports). Multiplying by a ±1 chipping sequence spreads signal energy
    uniformly across all frequencies, making subsequent time-domain subsampling
    equivalent to random Fourier sampling — which IS incoherent with wavelets
    and satisfies RIP (Candès et al., 2006).

    PHYSICAL INTERPRETATION: For ONT, the chipping sequence r could be applied
    as a known voltage modulation pattern before the ADC, making this approach
    physically implementable without hardware frequency-domain access.

    ADVANTAGE OVER OPTION A (Fourier subsampling): Measurements are real-valued.
    No complex arithmetic needed. Slightly simpler reconstruction.

    References:
        Tropp, Laska, Duarte, Romberg, Baraniuk (2010). Beyond Nyquist.
        Candès, Romberg, Tao (2006). Robust uncertainty principles.

    Args:
        n: Signal length
        m: Number of measurements (time-domain samples after modulation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (measure, adjoint, pseudo_inverse) callables
    """
    rng = np.random.default_rng(seed)
    # Rademacher chipping sequence: random ±1
    chipping = rng.choice([-1.0, 1.0], size=n)
    # Random subsampling indices (after modulation)
    indices = np.sort(rng.choice(n, size=m, replace=False))
    scale = np.sqrt(n / m)

    def measure(signal: np.ndarray) -> np.ndarray:
        """y = S · (r ⊙ x): modulate then subsample."""
        modulated = chipping * signal
        return modulated[indices] * scale

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        """phi^T(y) = r ⊙ S^T(y): upsample then demodulate.
        Since r ∈ {±1}, r^{-1} = r, so demodulation = modulation."""
        upsampled = np.zeros(n)
        upsampled[indices] = measurements * scale
        return chipping * upsampled

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        """Approximate pseudo-inverse (same as adjoint scaled by m/n).
        Note: not exact — a proper least-squares solve would be needed
        for exact inversion, but adjoint suffices for CoSaMP's proxy step."""
        upsampled = np.zeros(n)
        upsampled[indices] = measurements / scale
        return chipping * upsampled

    return measure, adjoint, pseudo_inverse


def create_wavelet_packet_operator(n: int, m: int, wavelet: str = 'haar',
                                    max_level: int = None, seed: int = None):
    """
    Creates a time-domain subsampling operator paired with a WP sparsity basis.

    Unlike Gaussian/Fourier subsampling where the measurement matrix drives
    incoherence, here incoherence is achieved by choosing a WP sparsity basis
    whose basis functions have lower pointwise amplitude (wider time support)
    than standard DWT wavelets.

    The measurement operator itself is plain time-domain subsampling S.
    The WP basis is selected via the Coifman-Wickerhauser best-basis algorithm
    (minimum Shannon entropy) and is stored in the returned ``wp_info`` dict
    for use by reconstruction algorithms.

    Theoretical basis
    -----------------
    WP basis functions at depth j satisfy:
        max_t |ψ_{j,p}(t)| ≤ C · 2^{-j/2}
    so the coherence with time-domain subsampling satisfies:
        μ(S, Ψ_WP at level j) ≤ C² · 2^{-j}
    which decreases with decomposition depth. The best-basis selection
    automatically finds the WP tree that minimises Shannon entropy of
    the signal's coefficients, trading coherence against sparsity.

    References
    ----------
    Coifman & Wickerhauser (1992). Entropy-based algorithms for best
        basis selection. IEEE Trans. Inf. Theory, 38(2), 713–718.
    Mallat (2008). A Wavelet Tour of Signal Processing (3rd ed.), Ch. 8.

    Args:
        n: Signal length (power of 2)
        m: Number of measurements
        wavelet: Orthogonal wavelet for WP decomposition (e.g. 'haar', 'db4')
        max_level: WP tree depth (default: log2(n))
        seed: Random seed for subsampling indices

    Returns:
        Tuple of (measure, adjoint, pseudo_inverse, wp_info) where wp_info is
        a dict with keys 'wavelet', 'max_level', 'n' (leaf_nodes are computed
        per-signal via best_basis_selection from wavelet_packet_module).
    """
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))
    scale = np.sqrt(n / m)

    if max_level is None:
        max_level = int(np.log2(n))

    def measure(signal: np.ndarray) -> np.ndarray:
        """Plain time-domain subsampling S·x (scaled)."""
        return signal[indices] * scale

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        """S^T: scatter measurements back to time domain (scaled)."""
        upsampled = np.zeros(n)
        upsampled[indices] = measurements * scale
        return upsampled

    def pseudo_inverse(measurements: np.ndarray) -> np.ndarray:
        """S^†: upsample scaled by sqrt(m/n)."""
        upsampled = np.zeros(n)
        upsampled[indices] = measurements * np.sqrt(m / n)
        return upsampled

    wp_info = {'wavelet': wavelet, 'max_level': max_level, 'n': n}
    return measure, adjoint, pseudo_inverse, wp_info


MEASUREMENT_OPERATORS = {
    "subsampling": create_subsampling_operator,
    "gaussian": create_gaussian_operator,
    "fourier_subsampling": create_fourier_subsampling_operator,
    "hadamard": create_hadamard_operator,
    "hadamard_multilevel": lambda n, m, seed=None: create_hadamard_multilevel_operator(
        n, m, seed=seed
    )[:3],
    "random_modulation": create_random_modulation_operator,
    "wavelet_packet": lambda n, m, seed=None: create_wavelet_packet_operator(n, m, seed=seed)[:3],
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


def create_measurement_operator(
    measurement_mode: str,
    n: int,
    m: int,
    seed: int | None = None,
):
    """Compatibility wrapper that always returns (measure, adjoint, pseudo_inverse)."""
    if measurement_mode == "gaussian":
        return _create_gaussian_operator_with_pinv(n, m, seed)

    operators = create_measurement_operators(measurement_mode, n, m, seed)
    if len(operators) == 3:
        return operators

    measure, adjoint = operators
    return measure, adjoint, adjoint
