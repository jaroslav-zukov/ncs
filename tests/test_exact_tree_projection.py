import pytest
from numpy.testing import assert_array_equal

from ncs.exact_tree_projection import tree_projection
from ncs.wt_coeffs import WtCoeffs


@pytest.mark.parametrize(
    "coeff_groups,tree_sparsity,expected_flat_coeffs",
    [
        ([[4], [1], [3, 2]], 3, [4, 1, 3, 0]),
        (
            [[1], [1], [2, 1], [5, 1, 4, 4], [1, 1, 1, 1, 1, 1, 1, 1]],
            5,
            [1, 1, 0, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ],
)
def test_tree_projection(coeff_groups, tree_sparsity, expected_flat_coeffs):
    wt_coeffs = WtCoeffs(coeff_groups, "haar")

    projected_coeffs = tree_projection(wt_coeffs, tree_sparsity)

    assert projected_coeffs.root_count == wt_coeffs.root_count
    assert projected_coeffs.max_level == wt_coeffs.max_level
    assert projected_coeffs.wavelet == wt_coeffs.wavelet
    assert_array_equal(projected_coeffs.flat_coeffs, expected_flat_coeffs)


def test_unsupported_root_size():
    wt_coeffs = WtCoeffs([[1, 2, 3, 4], [5, 6, 7, 8]], wavelet="db2")
    with pytest.raises(ValueError, match="Only root count 1 is supported"):
        tree_projection(wt_coeffs, 3)
