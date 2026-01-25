from typing import Callable

import numpy as np
from tqdm import tqdm

from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs


def cosamp_reconstruct(
    y: np.ndarray,
    tree_sparsity: int,
    x_init: WtCoeffs,
    measurement_op: Callable[[np.ndarray], np.ndarray],
    adjoint_op: Callable[[np.ndarray], np.ndarray],
) -> WtCoeffs:
    x_hat = x_init
    wavelet = x_init.wavelet
    r = np.copy(y)
    iteration_threshold = 50

    for _ in tqdm(range(iteration_threshold), desc="CoSaMP iterations", disable=True):
        upsampled_r = adjoint_op(r)
        e_coeffs = forward_transform(upsampled_r, wavelet)
        e_double_support = tree_projection(e_coeffs, 2 * tree_sparsity).support
        x_hat_support = x_hat.support

        t = x_hat_support | e_double_support

        coeffs_estimate = forward_transform(
            signal=adjoint_op(y), wavelet=wavelet
        ).on_support(t)

        x_hat = tree_projection(coeffs_estimate, tree_sparsity)
        r = y - measurement_op(inverse_transform(x_hat))
        # print(f"Iteration {i}\t Residue Mean: {np.mean(r ** 2):.1f}")

    return x_hat


MeasurementFunction = Callable[[np.ndarray], np.ndarray]
ReconstructionAlgorithm = Callable[
    [np.ndarray, int, WtCoeffs, MeasurementFunction, MeasurementFunction],
    WtCoeffs,
]

RECONSTRUCTION_ALGORITHMS: dict[str, ReconstructionAlgorithm] = {
    "CoSaMP": cosamp_reconstruct,
}


def reconstruct(
    reconstruction_mode: str,
    y: np.ndarray,
    x_init: WtCoeffs,
    tree_sparsity: int,
    measurement_op: Callable[[np.ndarray], np.ndarray],
    adjoint_op: Callable[[np.ndarray], np.ndarray],
):
    if reconstruction_mode not in RECONSTRUCTION_ALGORITHMS.keys():
        raise ValueError(
            f"Reconstruction mode {reconstruction_mode} not supported ({RECONSTRUCTION_ALGORITHMS.keys()})"
        )

    reconstruction_algorithm = RECONSTRUCTION_ALGORITHMS[reconstruction_mode]

    x_hat = reconstruction_algorithm(
        y, tree_sparsity, x_init, measurement_op, adjoint_op
    )

    return x_hat
