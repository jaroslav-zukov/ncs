import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.reconstruction_module import RECONSTRUCTION_ALGORITHMS, reconstruct
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs
from ncs.wt_coeffs import WtCoeffs


def test_reconstruction_validation(mocker):
    with pytest.raises(
        ValueError, match="Reconstruction mode invalid_reconstruction not supported"
    ):
        reconstruct(
            reconstruction_mode="invalid_reconstruction",
            y=np.array([1, 2, 3]),
            x_init=mocker.Mock(spec=WtCoeffs),
            tree_sparsity=1,
            compressive_sensing_operators=(lambda x: x, lambda x: x, lambda x: x),
        )


def test_reconstruction_calls_algorithm(mocker):
    mock_algorithm = mocker.Mock(return_value="reconstructed_result")
    reconstruction_algorithm_names = RECONSTRUCTION_ALGORITHMS.keys()
    mock_algorithms = {name: mock_algorithm for name in reconstruction_algorithm_names}

    mocker.patch(
        "ncs.reconstruction_module.RECONSTRUCTION_ALGORITHMS",
        mock_algorithms,
    )

    y = np.array([1, 2, 3])
    x_init = mocker.Mock(spec=WtCoeffs)
    tree_sparsity = 1
    measurement_op = lambda x: x
    adjoint_op = lambda x: x
    pseudo_inverse_op = lambda x: x
    compressive_sensing_operators = (measurement_op, adjoint_op, pseudo_inverse_op)

    result = reconstruct(
        reconstruction_mode="CoSaMP",
        y=y,
        x_init=x_init,
        tree_sparsity=tree_sparsity,
        compressive_sensing_operators=compressive_sensing_operators,
    )

    mock_algorithm.assert_called_once_with(
        y, tree_sparsity, x_init, compressive_sensing_operators
    )

    assert result == "reconstructed_result"


def test_cosamp_reconstruct_integration():
    tree_sparsity = 5
    seed = 123
    sparse_coeff = generate_tree_sparse_coeffs(
        power=7,
        count=1,
        tree_sparsity=tree_sparsity,
        wavelet="haar",
        seed=seed,
    )[0]

    x_hat = measure_and_reconstruct(
        measurement_mode="gaussian",
        m=100,
        reconstruction_mode="CoSaMP",
        coeffs_x=sparse_coeff,
        target_tree_sparsity=tree_sparsity,
        seed=seed,
    )

    assert x_hat.support == sparse_coeff.support
    assert_allclose(x_hat.flat_coeffs, sparse_coeff.flat_coeffs, atol=86)
