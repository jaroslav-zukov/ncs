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

    measurement_op, adjoint_op = create_measurement_operator(
        measurement_mode, n, m, seed
    )

    if measurement_mode == 'gaussian':
        y = measurement_op(coeffs_x.flat_coeffs)
    elif measurement_mode == 'subsampling':
        y = measurement_op(inverse_transform(coeffs_x))
        raw_measurement_op = measurement_op
        raw_adjoint_op = adjoint_op

        def composed_measurement_op(wt_flat: np.ndarray) -> np.ndarray:
            return raw_measurement_op(inverse_transform(
                WtCoeffs.from_flat_coeffs(wt_flat, coeffs_x.root_count, coeffs_x.max_level, coeffs_x.wavelet)
            ))

        def composed_adjoint_op(y_: np.ndarray) -> np.ndarray:
            return forward_transform(raw_adjoint_op(y_), coeffs_x.wavelet).flat_coeffs

        measurement_op = composed_measurement_op
        adjoint_op = composed_adjoint_op
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
        measurement_op=measurement_op,
        adjoint_op=adjoint_op,
    )

    return x_hat
