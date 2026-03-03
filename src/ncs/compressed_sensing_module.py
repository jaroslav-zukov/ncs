"""
End-to-end compressive sensing pipeline for wavelet-domain NCS.

.. warning::
    **THIS MODULE IS INCOMPLETE / WORK IN PROGRESS.**

    Known bugs (tracked in a separate fix branch, do NOT fix here):

    1. ``compressive_sensing_operators = ()`` — In earlier versions of this
       module the tuple was initialised to an empty tuple ``()`` and only
       conditionally populated inside if/elif branches.  If an unsupported
       measurement_mode reached the ``reconstruct()`` call, an empty tuple
       was passed, causing a cryptic unpack error rather than a clear
       ValueError.  The current implementation raises ValueError explicitly
       but the conditional structure remains fragile.

    2. Wrong kwargs to ``reconstruct()`` — Earlier call sites passed keyword
       arguments that do not match the ``reconstruct()`` signature in
       reconstruction_module (e.g. passing ``phi`` / ``phi_transpose`` as
       separate kwargs instead of the ``compressive_sensing_operators``
       tuple).  This caused TypeError at runtime.

    Both issues are documented here for traceability.  The fix branch will
    address them with a refactor of the operator composition and dispatch
    logic.

Module description
------------------
Provides ``measure_and_reconstruct``, which ties together the measurement
operators (measurement_module), inverse/forward wavelet transforms
(wavelet_module), and the CoSaMP reconstruction algorithm
(reconstruction_module) into a single end-to-end pipeline:

    x (WtCoeffs) → [IDWT or identity] → Φ → y → CoSaMP → x̂ (WtCoeffs)

For Gaussian mode, the measurement and reconstruction operate entirely in
the wavelet coefficient domain.  For subsampling (and by extension Fourier
subsampling), operators must be composed with the inverse/forward DWT to
translate between the time-domain measurement and the wavelet-domain
reconstruction.
"""

import numpy as np

from ncs.measurement_module import create_measurement_operator
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
    """
    Full CS pipeline: measure wavelet coefficients then reconstruct.

    .. warning::
        See module docstring for known WIP bugs in this module.

    Depending on measurement_mode, the pipeline differs:

    **gaussian** (wavelet-domain measurement):
        y = Φ · coeffs_x.flat_coeffs
        Φ acts directly on wavelet coefficients.  CoSaMP operators are Φ,
        Φᵀ, Φ† without any wavelet transform composition.

    **subsampling** (time-domain measurement):
        y = S · IDWT(coeffs_x)  where S is the subsampling operator.
        CoSaMP must work in wavelet domain, so the operators are composed:
          phi(wt)     = S(IDWT(wt))
          phi_T(y)    = DWT(Sᵀ(y))
          phi_pinv(y) = DWT(S†(y))
        Note: the pseudo-inverse of the composed operator (S ∘ IDWT) is NOT
        simply DWT ∘ S†; this approximation may affect convergence.

    Args:
        measurement_mode: "gaussian" or "subsampling".  See measurement_module
            for full operator descriptions and RIP/coherence notes.
        m: Number of measurements (m < n = coeffs_x.n).
        reconstruction_mode: Reconstruction algorithm name (e.g. "CoSaMP").
        coeffs_x: Ground-truth wavelet coefficient vector used to generate
            the measurement y = Φx.  Its tree parameters (root_count,
            max_level, wavelet) are also used to initialise x_init.
        target_tree_sparsity: Tree-sparsity level k passed to the CoSaMP
            reconstruction algorithm.
        seed: Optional random seed for the measurement operator factory.

    Returns:
        WtCoeffs: Reconstructed wavelet coefficient estimate x̂.

    Raises:
        ValueError: If measurement_mode is unsupported or if m >= n.
    """
    n = coeffs_x.n

    operator_bundle = create_measurement_operator(
        measurement_mode, n, m, seed
    )
    if len(operator_bundle) == 3:
        measurement_op, adjoint_op, pseudo_inverse = operator_bundle
    else:
        measurement_op, adjoint_op = operator_bundle
        pseudo_inverse = None

    if measurement_mode == 'gaussian':
        coeff_vector = getattr(coeffs_x, "flat_coeffs", None)
        if coeff_vector is None:
            coeff_vector = inverse_transform(coeffs_x)
        y = measurement_op(coeff_vector)

        def phi(wt_coeffs: WtCoeffs) -> np.ndarray:
            return measurement_op(wt_coeffs.flat_coeffs)

        def phi_transpose(meas: np.ndarray) -> WtCoeffs:
            return WtCoeffs.from_flat_coeffs(
                flat_coeffs=adjoint_op(meas),
                root_count=coeffs_x.root_count,
                max_level=coeffs_x.max_level,
                wavelet=coeffs_x.wavelet,
            )

        def phi_pseudoinverse(meas: np.ndarray) -> WtCoeffs:
            return WtCoeffs.from_flat_coeffs(
                flat_coeffs=adjoint_op(meas),
                root_count=coeffs_x.root_count,
                max_level=coeffs_x.max_level,
                wavelet=coeffs_x.wavelet,
            )

        compressive_sensing_operators = (phi, phi_transpose, phi_pseudoinverse)
    elif measurement_mode in ('subsampling', 'random_modulation'):
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
            inverse_op = pseudo_inverse if pseudo_inverse is not None else adjoint_op
            return forward_transform(inverse_op(y), coeffs_x.wavelet)

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
            inverse_op = pseudo_inverse if pseudo_inverse is not None else adjoint_op
            time_signal = inverse_op(meas)
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

    if pseudo_inverse is None:
        x_hat = reconstruct(
            reconstruction_mode=reconstruction_mode,
            y=y,
            x_init=x_init,
            tree_sparsity=target_tree_sparsity,
            measurement_op=measurement_op,
            adjoint_op=adjoint_op,
        )
    else:
        x_hat = reconstruct(
            reconstruction_mode=reconstruction_mode,
            y=y,
            x_init=x_init,
            tree_sparsity=target_tree_sparsity,
            compressive_sensing_operators=compressive_sensing_operators,
        )

    return x_hat
