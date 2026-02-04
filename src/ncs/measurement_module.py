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
    if m >= n:
        raise ValueError("m must be less than n")

    if measurement_mode not in MEASUREMENT_OPERATORS.keys():
        raise ValueError(
            f"Measurement mode {measurement_mode} is not supported ({MEASUREMENT_OPERATORS.keys()})"
        )

    measurement_operators = MEASUREMENT_OPERATORS[measurement_mode]
    return measurement_operators(n, m, seed)
