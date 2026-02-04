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
    compressive_sensing_operators,
) -> WtCoeffs:
    x_hat = x_init
    r = np.copy(y)
    iteration_threshold = 20

    (phi, phi_transpose, phi_pseudoinverse) = compressive_sensing_operators

    for _ in tqdm(range(iteration_threshold), desc="CoSaMP iterations", disable=True):
        e = phi_transpose(r)
        omega_e_double_support = tree_projection(e, 2 * tree_sparsity).support
        t = x_hat.support | omega_e_double_support
        b_coeffs_estimate = phi_pseudoinverse(y).on_support(t)
        x_hat = tree_projection(b_coeffs_estimate, tree_sparsity)
        r = y - phi(x_hat)
    return x_hat


# TODO: Update types
# MeasurementFunction = Callable[[np.ndarray], np.ndarray]
# ReconstructionAlgorithm = Callable[
#     [np.ndarray, int, WtCoeffs, MeasurementFunction, MeasurementFunction],
#     WtCoeffs,
# ]

# RECONSTRUCTION_ALGORITHMS: dict[str, ReconstructionAlgorithm] = {
RECONSTRUCTION_ALGORITHMS = {
    "CoSaMP": cosamp_reconstruct,
}


def reconstruct(
    reconstruction_mode: str,
    y: np.ndarray,
    x_init: WtCoeffs,
    tree_sparsity: int,
    compressive_sensing_operators,
):
    if reconstruction_mode not in RECONSTRUCTION_ALGORITHMS.keys():
        raise ValueError(
            f"Reconstruction mode {reconstruction_mode} not supported ({RECONSTRUCTION_ALGORITHMS.keys()})"
        )

    reconstruction_algorithm = RECONSTRUCTION_ALGORITHMS[reconstruction_mode]

    x_hat = reconstruction_algorithm(
        y, tree_sparsity, x_init, compressive_sensing_operators
    )

    return x_hat
