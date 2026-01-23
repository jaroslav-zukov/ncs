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
        subsample_op: Callable[[np.ndarray], np.ndarray],
        upsample_op: Callable[[np.ndarray], np.ndarray],
) -> WtCoeffs:
    x_hat = x_init
    wavelet = x_init.wavelet
    r = np.copy(y)
    iteration_threshold = 100

    for _ in tqdm(range(iteration_threshold), desc="CoSaMP iterations"):
        upsampled_r = upsample_op(r)
        e_coeffs = forward_transform(upsampled_r, wavelet)
        e_double_support = tree_projection(e_coeffs, 2 * tree_sparsity).support

        t = x_hat.support | e_double_support

        coeffs_estimate = forward_transform(
            signal=upsample_op(y),
            wavelet=wavelet
        ).on_support(t)

        x_hat = tree_projection(coeffs_estimate, tree_sparsity)
        r = y - subsample_op(inverse_transform(x_hat))

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


def measure_and_reconstruct(
        measurement_mode: str,
        m: int,
        reconstruction_mode: str,
        coeffs_x: WtCoeffs,
        target_tree_sparsity: int,
) -> WtCoeffs:
    supported_measurement_modes = ['subsampling']

    if measurement_mode not in supported_measurement_modes:
        raise ValueError(f"Measurement mode {measurement_mode} is not supported ({supported_measurement_modes})")

    signal_z = inverse_transform(coeffs_x)
    n = coeffs_x.n
    wavelet = coeffs_x.wavelet

    # TODO: implement some real choice switch case
    subsample_op, upsample_op = create_subsampling_operator(n=n, m=m)

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
    supported_reconstruction_modes = ['CoSaMP']

    if reconstruction_mode not in supported_reconstruction_modes:
        raise ValueError(f"Reconstruction mode {reconstruction_mode} not supported ({supported_reconstruction_modes})")

    reconstruction_algorithm = RECONSTRUCTION_ALGORITHMS[reconstruction_mode]

    x_hat = reconstruction_algorithm(y, tree_sparsity, x_init, subsample_op, upsample_op)

    return x_hat


def create_subsampling_operator(n: int, m: int, seed: int = None):
    """
        Returns:
        A tuple of (subsample_fn, upsample_fn)
    """
    if m >= n:
        raise ValueError("m must be less than n")

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=m, replace=False))

    def subsample(signal: np.ndarray) -> np.ndarray:
        return signal[indices]

    def upsample(measurements: np.ndarray) -> np.ndarray:
        upsampled = np.zeros(n)
        upsampled[indices] = measurements
        return upsampled

    return subsample, upsample
