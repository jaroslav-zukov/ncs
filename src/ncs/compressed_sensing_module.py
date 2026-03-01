import numpy as np

from ncs.measurement_module import create_measurement_operators
from ncs.reconstruction_module import reconstruct
from ncs.wavelet_module import forward_transform, inverse_transform
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
        # Gaussian operator works directly in wavelet coefficient domain
        compressive_sensing_operators = (measurement_op, adjoint_op, pseudo_inverse)
    elif measurement_mode == 'subsampling':
        y = measurement_op(inverse_transform(coeffs_x))
        # For subsampling, the raw operators act in the time domain.
        # CoSaMP works in wavelet coefficient domain, so we compose the operators:
        #   phi(wt_coeffs)       = subsample(IDWT(wt_coeffs))
        #   phi_T(y)             = DWT(upsample(y))
        #   phi_pinv(y)          = DWT(pseudo_inverse_subsample(y))
        # TODO: The pseudo-inverse of the composed operator (S ∘ IDWT) is NOT simply
        #   DWT ∘ S†. A proper least-squares pseudo-inverse may be needed for
        #   CoSaMP convergence guarantees, especially for the subsampling case.
        def phi(wt_coeffs: WtCoeffs) -> np.ndarray:
            return measurement_op(inverse_transform(wt_coeffs))

        def phi_transpose(y: np.ndarray) -> WtCoeffs:
            return forward_transform(adjoint_op(y), coeffs_x.wavelet)

        def phi_pseudoinverse(y: np.ndarray) -> WtCoeffs:
            return forward_transform(pseudo_inverse(y), coeffs_x.wavelet)

        compressive_sensing_operators = (phi, phi_transpose, phi_pseudoinverse)
    elif measurement_mode == 'fourier_subsampling':
        y = measurement_op(inverse_transform(coeffs_x))
        # For Fourier subsampling, the raw operators act in the frequency domain.
        # CoSaMP works in wavelet coefficient domain, so we compose the operators:
        #   phi(wt_coeffs)   = S · DFT · IDWT(wt_coeffs)
        #   phi_T(y)         = DWT · DFT^T · S^T(y)   (real adjoint of subsampled rfft)
        #   phi_pinv(y)      = same as phi_T (DFT is unitary with ortho norm)
        # This composition is incoherent with the wavelet basis, providing RIP
        # guarantees for wavelet-sparse signals (Candès et al., 2006).
        def phi(wt_coeffs: WtCoeffs) -> np.ndarray:
            return measurement_op(inverse_transform(wt_coeffs))

        def phi_transpose(meas: np.ndarray) -> WtCoeffs:
            time_signal = adjoint_op(meas)
            return forward_transform(time_signal, coeffs_x.wavelet)

        def phi_pseudoinverse(meas: np.ndarray) -> WtCoeffs:
            time_signal = pseudo_inverse(meas)
            return forward_transform(time_signal, coeffs_x.wavelet)

        compressive_sensing_operators = (phi, phi_transpose, phi_pseudoinverse)
    else:
        raise ValueError(f"Unknown measurement mode: {measurement_mode}")

    x_init = WtCoeffs.from_flat_coeffs(
        flat_coeffs=np.zeros(n),
        root_count=coeffs_x.root_count,
        max_level=coeffs_x.max_level,
        wavelet=coeffs_x.wavelet,
    )

    x_hat = reconstruct(
        reconstruction_mode=reconstruction_mode,
        y=y,
        x_init=x_init,
        tree_sparsity=target_tree_sparsity,
        compressive_sensing_operators=compressive_sensing_operators,
    )

    return x_hat
