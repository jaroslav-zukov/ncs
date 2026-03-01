"""
Discrete wavelet transform utilities for NCS.

This module wraps PyWavelets (pywt) to provide the forward (signal →
coefficients) and inverse (coefficients → signal) discrete wavelet
transforms used throughout the NCS pipeline.

Design constraints
------------------
orthogonality
    Only orthogonal wavelets are permitted.  Orthogonality ensures that the
    wavelet transform matrix Ψ satisfies ΨᵀΨ = I, which is essential for
    the RIP-based recovery guarantees used in compressive sensing.  Bi-
    orthogonal or non-orthogonal wavelets would require separate handling of
    the synthesis and analysis operators.

periodization mode
    pywt.wavedec / pywt.waverec are called with mode="periodization".  This
    mode treats the signal as periodic and produces coefficient arrays whose
    lengths sum to exactly n (the signal length) at every decomposition
    level, preserving the vector-space isomorphism ℝⁿ ↔ ℝⁿ required for
    the flat-coefficient representation used by WtCoeffs.  This also
    guarantees power-of-2 coefficient counts at each level when n is a
    power of 2, which is a requirement for the dyadic tree structure.

full decomposition
    level=None in pywt.wavedec causes decomposition to the maximum possible
    level, fully populating the wavelet tree from coarse (approximation)
    coefficients down to the finest detail level.
"""

import numpy as np
import pywt

from ncs.wt_coeffs import WtCoeffs


def get_orthogonal_wavelets():
    """
    Return the list of orthogonal discrete wavelets supported by PyWavelets.

    Queries pywt.wavelist for all discrete wavelets and filters to those
    flagged as orthogonal by pywt.Wavelet.orthogonal.  Common examples
    include 'haar', 'db2' … 'db38', 'sym2' … 'sym20', 'coif1' … 'coif17'.

    Returns:
        list[str]: Names of orthogonal wavelets, usable as the `wavelet`
            argument to forward_transform.
    """
    wavelets = pywt.wavelist(kind="discrete")
    orthogonal_wavelets = [w for w in wavelets if pywt.Wavelet(w).orthogonal]
    return orthogonal_wavelets


# Todo: add FFT support (if needed)
def forward_transform(signal: np.ndarray, wavelet):
    """
    Apply the forward discrete wavelet transform to a signal.

    Decomposes the signal into hierarchical wavelet coefficients using a
    fully recursive (maximum-level) dyadic DWT with periodization boundary
    conditions.  The result is wrapped in a WtCoeffs object that tracks the
    tree structure.

    The transform is orthogonal (for orthogonal wavelets), so the inverse
    is the exact adjoint: inverse_transform ∘ forward_transform = identity.

    Args:
        signal: 1-D numpy array of length n.  For the wavelet tree to have
            the expected dyadic structure, n should be a power of 2.
        wavelet: Name of an orthogonal wavelet (e.g. 'haar', 'db4').
            Must be in get_orthogonal_wavelets().

    Returns:
        WtCoeffs: Wavelet coefficient object with coeff_groups matching the
            pywt.wavedec output (coarsest to finest), max_level set to the
            number of decomposition levels, and root_count equal to the
            number of approximation (scaling) coefficients at the coarsest
            level.

    Raises:
        ValueError: If wavelet is not in the list of supported orthogonal
            wavelets.
    """
    supported_wavelets = get_orthogonal_wavelets()
    if wavelet not in supported_wavelets:
        raise ValueError(
            f"Wavelet '{wavelet}' not supported. Supported orthogonal wavelets: {supported_wavelets}"
        )

    wt_coeff_groups = pywt.wavedec(
        signal, wavelet=wavelet, level=None, mode="periodization"
    )
    return WtCoeffs(wt_coeff_groups, wavelet)


def inverse_transform(wt_coeffs: WtCoeffs) -> np.ndarray:
    """
    Apply the inverse discrete wavelet transform to reconstruct a signal.

    Synthesises a time-domain signal from a WtCoeffs coefficient object
    using the same periodization boundary mode as forward_transform, ensuring
    exact inversion for orthogonal wavelets.

    Args:
        wt_coeffs: WtCoeffs object containing grouped wavelet coefficients
            (coarsest to finest) and the wavelet name.

    Returns:
        np.ndarray: Reconstructed 1-D signal of length n = wt_coeffs.n.
            For orthogonal wavelets, forward_transform(inverse_transform(c))
            recovers c exactly (up to floating-point precision).
    """
    return pywt.waverec(wt_coeffs.coeff_groups, wt_coeffs.wavelet, mode="periodization")
