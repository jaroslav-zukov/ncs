from typing import Callable

import numpy as np


def create_subsampling_operator(n: int, m: int, seed: int = None):
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))

    # Adding 1 / np.sqrt(m) to subsample hasn't changed the behavior
    def subsample(signal: np.ndarray) -> np.ndarray:
        return signal[indices]

    def upsample(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled

    return subsample, upsample


def create_gaussian_operator(n: int, m: int, seed: int = None):
    rng = np.random.default_rng(seed)
    # Normalize variance by sqrt(1/m) so that E[|y|^2] = |x|^2
    phi = rng.normal(0, 1.0 / np.sqrt(m), size=(m, n))

    def measure(signal: np.ndarray) -> np.ndarray:
        return phi @ signal

    def adjoint(measurements: np.ndarray) -> np.ndarray:
        return phi.T @ measurements

    return measure, adjoint


MeasurementFunction = Callable[[np.ndarray], np.ndarray]
MeasurementOperators = Callable[
    [int, int, int | None], tuple[MeasurementFunction, MeasurementFunction]
]

MEASUREMENT_OPERATORS: dict[str, MeasurementOperators] = {
    "subsampling": create_subsampling_operator,
    "gaussian": create_gaussian_operator,
}


def create_measurement_operator(
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
