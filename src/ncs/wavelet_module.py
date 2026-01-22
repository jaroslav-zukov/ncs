import pywt

from src.ncs.wt_coeffs import WtCoeffs


def get_orthogonal_wavelets():
    wavelets = pywt.wavelist(kind="discrete")
    orthogonal_wavelets = [w for w in wavelets if pywt.Wavelet(w).orthogonal]
    return orthogonal_wavelets


# Todo: add FFT support (if needed)
def transform(signal, wavelet):
    supported_wavelets = get_orthogonal_wavelets()
    if wavelet not in supported_wavelets:
        raise ValueError(
            f"Wavelet '{wavelet}' not supported. Supported orthogonal wavelets: {supported_wavelets}"
        )

    wt_coeff_groups = pywt.wavedec(
        signal, wavelet=wavelet, level=None, mode="periodization"
    )
    return WtCoeffs(wt_coeff_groups)
