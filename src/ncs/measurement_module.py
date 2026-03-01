from typing import Callable

import numpy as np


def create_subsampling_operator(n: int, m: int, seed: int = None):
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


MEASUREMENT_OPERATORS = {
    "subsampling": create_subsampling_operator,
    "gaussian": create_gaussian_operator,
    "random_modulation": create_random_modulation_operator,
}


def create_measurement_operators(
    measurement_mode: str,
    n: int,
    m: int,
    seed: int | None = None,
):
    if m >= n:
        raise ValueError("m must be less than n")

    if measurement_mode not in MEASUREMENT_OPERATORS.keys():
        raise ValueError(
            f"Measurement mode {measurement_mode} is not supported ({MEASUREMENT_OPERATORS.keys()})"
        )

    measurement_operators = MEASUREMENT_OPERATORS[measurement_mode]
    return measurement_operators(n, m, seed)
