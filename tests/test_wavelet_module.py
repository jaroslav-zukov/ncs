import numpy as np
import pytest
from numpy.testing import assert_allclose

from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs


def test_forward_transform():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = forward_transform(signal, "db2")

    assert result.wavelet == "db2"
    assert result.root_count == 4
    assert result.max_level == 1
    assert_allclose(
        actual=result.flat_coeffs,
        desired=[4.8, 3.7, 6.6, 10.4, -1.0, 0, 0, 3.9],
        atol=0.05,
    )


def test_inverse_transform():
    flat_coeffs = [4.8, 3.7, 6.6, 10.4, -1.0, 0, 0, 3.9]
    wt_coeffs = WtCoeffs.from_flat_coeffs(
        flat_coeffs=flat_coeffs,
        root_count=4,
        max_level=1,
        wavelet="db2",
    )
    result = inverse_transform(wt_coeffs)
    assert_allclose(
        actual=result,
        desired=[1, 2, 3, 4, 5, 6, 7, 8],
        atol=0.05,
    )


@pytest.mark.parametrize("wavelet", ["db2", "db4", "haar", "sym2", "coif1"])
def test_forward_transform_orthogonal_wavelets_success(wavelet):
    signal = np.arange(32)
    result = forward_transform(signal, wavelet)
    assert isinstance(result, WtCoeffs)


@pytest.mark.parametrize(
    "wavelet,expected_error",
    [
        ("bior1.3", ValueError),
        ("bior2.2", ValueError),
        ("invalid_wavelet", ValueError),
        ("not_a_wavelet", ValueError),
    ],
)
def test_forward_transform_invalid_wavelets_failure(wavelet, expected_error):
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(expected_error):
        forward_transform(signal, wavelet)
