import pytest
from numpy import array, testing

from src.ncs.wt_coeffs import WtCoeffs


def test_init_empty_coeff_groups():
    with pytest.raises(ValueError, match="coeff_groups cannot be empty"):
        WtCoeffs(coeff_groups=[], wavelet="haar")


def test_init_non_list_coeff_groups():
    with pytest.raises(TypeError, match="coeff_groups must be a list"):
        WtCoeffs(coeff_groups="not a list", wavelet="haar")


def test_init_invalid_leaf_count():
    # root_count=1, max_level=2 expects 1*2^(2-1)=2 leaves, but we have 3
    with pytest.raises(ValueError, match="Invalid leaf count at level 2"):
        WtCoeffs(coeff_groups=[[1], [2], [3, 4, 5]], wavelet="haar")

    # Verify the error message includes the suggestion about periodization
    with pytest.raises(
        ValueError, match="Try setting pywt.wavedec mode='periodization'"
    ):
        WtCoeffs(coeff_groups=[[1], [2], [3, 4, 5, 6]], wavelet="haar")


def test_init_valid_structure():
    # Valid structure: root_count=1, max_level=2, expects 2 leaves
    wt_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    assert wt_coeffs.max_level == 2
    assert wt_coeffs.root_count == 1
    assert wt_coeffs.wavelet == "haar"


def test_multi_root_structure():
    coeff_groups = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    ]
    wt_coeffs = WtCoeffs(coeff_groups=coeff_groups, wavelet="db2")
    assert wt_coeffs.max_level == 3
    assert wt_coeffs.root_count == 4
    assert wt_coeffs.wavelet == "db2"
    assert wt_coeffs.coeff_groups == coeff_groups


@pytest.mark.parametrize(
    "coeff_groups,expected_flat",
    [
        # Single root: root_count=1, max_level=2
        ([[1], [2], [3, 4]], list(range(1, 5))),
        # Multiple roots: root_count=2, max_level=2
        ([[1, 2], [3, 4], [5, 6, 7, 8]], list(range(1, 9))),
        # Multiple roots: root_count=4, max_level=3
        (
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            ],
            list(range(1, 33)),
        ),
    ],
)
def test_flat_coeffs(coeff_groups, expected_flat):
    wt_coeffs = WtCoeffs(coeff_groups=coeff_groups, wavelet="haar")
    testing.assert_array_equal(wt_coeffs.flat_coeffs, expected_flat)
    assert len(wt_coeffs.flat_coeffs) == len(expected_flat)


def test_support_property():
    # Test with zeros and non-zeros
    wt_coeffs = WtCoeffs(coeff_groups=[[0], [2], [0, 4]], wavelet="haar")
    assert wt_coeffs.support == {1, 3}

    # Test with all non-zero
    wt_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    assert wt_coeffs.support == {0, 1, 2, 3}

    # Test with all zeros
    wt_coeffs = WtCoeffs(coeff_groups=[[0], [0], [0, 0]], wavelet="haar")
    assert wt_coeffs.support == set()


def test_n_property():
    # Single root structure
    wt_coeffs = WtCoeffs(coeff_groups=[[0], [0], [3, 4]], wavelet="haar")
    assert wt_coeffs.n == 4

    # Multiple roots structure
    wt_coeffs = WtCoeffs(coeff_groups=[[1, 2], [3, 4], [5, 6, 7, 8]], wavelet="haar")
    assert wt_coeffs.n == 8

    # Larger structure
    wt_coeffs = WtCoeffs(
        coeff_groups=[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        ],
        wavelet="db2",
    )
    assert wt_coeffs.n == 32


def test_equality_operator():
    # Base case for comparison
    base_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")

    # Test equality with identical instance
    identical_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    assert base_coeffs == identical_coeffs

    # Test inequality with different type
    assert base_coeffs != "not a WtCoeffs object"
    assert base_coeffs != [[1], [2], [3, 4]]
    assert base_coeffs is not None

    # Test inequality with different root_count and max level
    different_root_count = WtCoeffs(coeff_groups=[[1, 2], [3, 4]], wavelet="haar")
    assert base_coeffs != different_root_count

    # Test inequality with different coefficients in first group
    different_first_group = WtCoeffs(coeff_groups=[[999], [2], [3, 4]], wavelet="haar")
    assert base_coeffs != different_first_group

    # Test inequality with different coefficients in middle group
    different_middle_group = WtCoeffs(coeff_groups=[[1], [999], [3, 4]], wavelet="haar")
    assert base_coeffs != different_middle_group

    # Test inequality with different coefficients in last group
    different_last_group = WtCoeffs(coeff_groups=[[1], [2], [999, 4]], wavelet="haar")
    assert base_coeffs != different_last_group

    different_last_group_2 = WtCoeffs(coeff_groups=[[1], [2], [3, 999]], wavelet="haar")
    assert base_coeffs != different_last_group_2


def test_from_flat_coeffs():
    # Test single root structure
    flat = [1, 2, 3, 4]
    wt_coeffs = WtCoeffs.from_flat_coeffs(
        flat, root_count=1, max_level=2, wavelet="haar"
    )

    expected_groups = [[1], [2], [3, 4]]
    for i, expected in enumerate(expected_groups):
        testing.assert_array_equal(wt_coeffs.coeff_groups[i], expected)
    assert wt_coeffs.max_level == 2
    assert wt_coeffs.root_count == 1
    testing.assert_array_equal(wt_coeffs.flat_coeffs, flat)

    # Test multiple roots structure
    flat = [1, 2, 3, 4, 5, 6, 7, 8]
    wt_coeffs = WtCoeffs.from_flat_coeffs(
        flat, root_count=2, max_level=2, wavelet="haar"
    )

    expected_groups = [[1, 2], [3, 4], [5, 6, 7, 8]]
    for i, expected in enumerate(expected_groups):
        testing.assert_array_equal(wt_coeffs.coeff_groups[i], expected)
    assert wt_coeffs.max_level == 2
    assert wt_coeffs.root_count == 2
    testing.assert_array_equal(wt_coeffs.flat_coeffs, flat)

    # Test larger structure
    flat = list(range(1, 33))
    wt_coeffs = WtCoeffs.from_flat_coeffs(
        flat, root_count=4, max_level=3, wavelet="db2"
    )

    expected_groups = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    ]
    for i, expected in enumerate(expected_groups):
        testing.assert_array_equal(wt_coeffs.coeff_groups[i], expected)
    assert wt_coeffs.wavelet == "db2"

    # Test with numpy array
    flat = array([1.5, 2.5, 3.5, 4.5])
    wt_coeffs = WtCoeffs.from_flat_coeffs(
        flat, root_count=1, max_level=2, wavelet="haar"
    )
    testing.assert_array_equal(wt_coeffs.flat_coeffs, flat)

    # Test invalid length
    with pytest.raises(ValueError, match="Invalid flat_coeffs length"):
        WtCoeffs.from_flat_coeffs([1, 2, 3], root_count=1, max_level=2, wavelet="haar")


def test_on_support():
    # Test with partial support
    wt_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    support = {0, 2}
    result = wt_coeffs.on_support(support)

    expected = WtCoeffs(coeff_groups=[[1], [0], [3, 0]], wavelet="haar")
    assert result == expected
    assert result.support == support

    # Test with empty support
    wt_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    result = wt_coeffs.on_support(set())

    expected = WtCoeffs(coeff_groups=[[0], [0], [0, 0]], wavelet="haar")
    assert result == expected
    assert result.support == set()

    # Test with full support
    wt_coeffs = WtCoeffs(coeff_groups=[[1], [2], [3, 4]], wavelet="haar")
    support = {0, 1, 2, 3}
    result = wt_coeffs.on_support(support)

    assert result == wt_coeffs
    assert result.support == support

    # Test with larger structure
    wt_coeffs = WtCoeffs(coeff_groups=[[1, 2], [3, 4], [5, 6, 7, 8]], wavelet="haar")
    support = {1, 3, 5, 7}
    result = wt_coeffs.on_support(support)

    expected = WtCoeffs(coeff_groups=[[0, 2], [0, 4], [0, 6, 0, 8]], wavelet="haar")
    assert result == expected
    assert result.support == support

    # Test immutability of original instance - on support returns new instance
    original_flat = wt_coeffs.flat_coeffs.copy()
    wt_coeffs.on_support({0})
    testing.assert_array_equal(wt_coeffs.flat_coeffs, original_flat)
