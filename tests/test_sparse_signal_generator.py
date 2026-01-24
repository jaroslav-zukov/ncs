from numpy.testing import assert_allclose

from src.ncs.sparse_signal_generator import (
    generate_tree_sparse_coeffs,
    generate_tree_sparse_signals,
)


def test_generate_sparse_coeffs():
    result = generate_tree_sparse_coeffs(
        power=3,
        count=2,
        tree_sparsity=4,
        wavelet="haar",
        seed=123,
    )
    expected = [
        [-1.7, 269.7, 0, -293, 0, 0, 0, 345.8],
        [-219.5, 79.5, 0, -228.5, 0, 0, 0, -333.0],
    ]

    assert len(result) == 2
    for i in range(2):
        assert_allclose(result[i].flat_coeffs, expected[i], atol=1e-1)


def test_generate_tree_sparse_signals():
    result = generate_tree_sparse_signals(
        power=3,
        count=1,
        tree_sparsity=4,
        wavelet="haar",
        seed=123,
    )[0]

    expected = [94.7, 94.7, 94.7, 94.7, -242.5, -242.5, 295, -194]
    assert_allclose(result, expected, atol=1e-1)
