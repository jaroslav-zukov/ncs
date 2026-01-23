from typing import Callable

import numpy as np
from tqdm import tqdm

from src.ncs.exact_tree_projection import tree_projection
from src.ncs.wavelet_module import forward_transform, inverse_transform
from src.ncs.wt_coeffs import WtCoeffs


def cosamp_reconstruct(
        y: np.ndarray,
        tree_sparsity: int,
        x_init: WtCoeffs,
        measure_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
) -> WtCoeffs:
    x_hat = x_init
    wavelet = x_init.wavelet
    r = np.copy(y)
    iteration_threshold = 50

    for i in tqdm(range(iteration_threshold), desc="CoSaMP iterations"):
        upsampled_r = adjoint_op(r)
        e_coeffs = forward_transform(upsampled_r, wavelet)
        e_double_support = tree_projection(e_coeffs, 2 * tree_sparsity).support
        x_hat_support = x_hat.support

        t = x_hat_support | e_double_support

        coeffs_estimate = forward_transform(
            signal=adjoint_op(y),
            wavelet=wavelet
        ).on_support(t)

        x_hat = tree_projection(coeffs_estimate, tree_sparsity)
        r = y - measure_op(inverse_transform(x_hat))
        # print(f"Iteration {i}\t Residue Mean: {np.mean(r ** 2):.1f}")

    return x_hat


ReconstructionAlgorithm = Callable[
    [np.ndarray,
     int,
     WtCoeffs,
     Callable[[np.ndarray], np.ndarray],
     Callable[[np.ndarray], np.ndarray]
     ],
    WtCoeffs
]

RECONSTRUCTION_ALGORITHMS: dict[str, ReconstructionAlgorithm] = {
    'CoSaMP': cosamp_reconstruct,
}


def create_subsampling_operator(n: int, m: int, seed: int = None):
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))

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


def create_measurement_operator(
        measurement_mode: str,
        n: int,
        m: int,
):
    if m >= n:
        raise ValueError("m must be less than n")

    supported_measurement_modes = ['subsampling', 'gaussian']

    if measurement_mode not in supported_measurement_modes:
        raise ValueError(f"Measurement mode {measurement_mode} is not supported ({supported_measurement_modes})")

    if measurement_mode == 'subsampling':
        return create_subsampling_operator(n=n, m=m)
    elif measurement_mode == 'gaussian':
        return create_gaussian_operator(n=n, m=m)
    else:
        raise ValueError(f"Unknown measurement mode {measurement_mode}")


def measure_and_reconstruct(
        measurement_mode: str,
        m: int,
        reconstruction_mode: str,
        coeffs_x: WtCoeffs,
        target_tree_sparsity: int,
) -> WtCoeffs:
    signal_z = inverse_transform(coeffs_x)
    n = coeffs_x.n
    wavelet = coeffs_x.wavelet

    # TODO: implement some real choice switch case
    subsample_op, upsample_op = create_measurement_operator(
        measurement_mode=measurement_mode,
        n=n,
        m=m
    )

    y = subsample_op(signal_z)

    x_init = WtCoeffs.from_flat_coeffs(
        flat_coeffs=np.zeros(n),
        root_count=coeffs_x.root_count,
        max_level=coeffs_x.max_level,
        wavelet=wavelet,
    )

    x_hat = reconstruct(reconstruction_mode, y, x_init, target_tree_sparsity, subsample_op, upsample_op)

    return x_hat


def reconstruct(
        reconstruction_mode: str,
        y: np.ndarray,
        x_init: WtCoeffs,
        tree_sparsity: int,
        subsample_op: Callable[[np.ndarray], np.ndarray],
        upsample_op: Callable[[np.ndarray], np.ndarray],
):
    if reconstruction_mode not in RECONSTRUCTION_ALGORITHMS.keys():
        raise ValueError(
            f"Reconstruction mode {reconstruction_mode} not supported ({RECONSTRUCTION_ALGORITHMS.keys()})")

    reconstruction_algorithm = RECONSTRUCTION_ALGORITHMS[reconstruction_mode]

    x_hat = reconstruction_algorithm(y, tree_sparsity, x_init, subsample_op, upsample_op)

    return x_hat
