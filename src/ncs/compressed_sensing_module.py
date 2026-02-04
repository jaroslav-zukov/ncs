import numpy as np

from ncs.measurement_module import create_measurement_operators
from ncs.reconstruction_module import reconstruct
from ncs.wavelet_module import inverse_transform
from ncs.wt_coeffs import WtCoeffs


def measure_and_reconstruct(
    measurement_mode: str,
    m: int,
    reconstruction_mode: str,
    coeffs_x: WtCoeffs,
    target_tree_sparsity: int,
    seed: int | None = None,
) -> WtCoeffs:
    n = coeffs_x.n

    measurement_op, adjoint_op, pseudo_inverse = create_measurement_operators(
        measurement_mode, n, m, seed
    )

    if measurement_mode == 'gaussian':
        y = measurement_op(coeffs_x)
    elif measurement_mode == 'subsampling':
        y = measurement_op(inverse_transform(coeffs_x))
    else:
        raise ValueError(f"Unknown measurement mode: {measurement_mode}")


    x_init = WtCoeffs.from_flat_coeffs(
        flat_coeffs=np.zeros(n),
        root_count=coeffs_x.root_count,
        max_level=coeffs_x.max_level,
        wavelet=coeffs_x.wavelet,
    )



    compressive_sensing_operators = ()

    x_hat = reconstruct(
        reconstruction_mode=reconstruction_mode,
        y=y,
        x_init=x_init,
        tree_sparsity=target_tree_sparsity,
        measurement_op=measurement_op,
        adjoint_op=adjoint_op,
    )

    return x_hat
