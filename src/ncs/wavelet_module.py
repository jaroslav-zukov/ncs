import numpy as np
import pywt

from ncs.wt_coeffs import WtCoeffs


def get_orthogonal_wavelets():
    wavelets = pywt.wavelist(kind="discrete")
    orthogonal_wavelets = [w for w in wavelets if pywt.Wavelet(w).orthogonal]
    return orthogonal_wavelets


# Todo: add FFT support (if needed)
def forward_transform(signal: np.ndarray, wavelet):
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
    return pywt.waverec(wt_coeffs.coeff_groups, wt_coeffs.wavelet, mode="periodization")
