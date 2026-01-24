import numpy as np

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.wt_coeffs import WtCoeffs


def test_measure_and_reconstruct(mocker):
    measurement_mode = "gaussian"
    m = 50
    reconstruction_mode = "tree_sparse"
    target_tree_sparsity = 10
    n = 1000

    coeffs_x = mocker.Mock(spec=WtCoeffs)
    coeffs_x.n = n
    coeffs_x.root_count = 1
    coeffs_x.max_level = 3
    coeffs_x.wavelet = "db4"

    mock_signal_z = np.random.randn(n)
    mock_measurement_op = mocker.Mock(return_value=np.random.randn(m))
    mock_adjoint_op = mocker.Mock()
    mock_x_hat = mocker.Mock(spec=WtCoeffs)

    mocker.patch("ncs.compressed_sensing_module.inverse_transform", return_value=mock_signal_z)
    mock_create_meas = mocker.patch("ncs.compressed_sensing_module.create_measurement_operator", return_value=(mock_measurement_op, mock_adjoint_op))
    mocker.patch("ncs.compressed_sensing_module.WtCoeffs.from_flat_coeffs")
    mock_reconstruct = mocker.patch("ncs.compressed_sensing_module.reconstruct", return_value=mock_x_hat)

    result = measure_and_reconstruct(
        measurement_mode=measurement_mode,
        m=m,
        reconstruction_mode=reconstruction_mode,
        coeffs_x=coeffs_x,
        target_tree_sparsity=target_tree_sparsity,
    )

    mock_create_meas.assert_called_once_with(measurement_mode, n, m)

    mock_reconstruct.assert_called_once()
    call_kwargs = mock_reconstruct.call_args.kwargs
    assert call_kwargs["reconstruction_mode"] == reconstruction_mode
    assert call_kwargs["tree_sparsity"] == target_tree_sparsity
    assert call_kwargs["measurement_op"] == mock_measurement_op
    assert call_kwargs["adjoint_op"] == mock_adjoint_op

    assert result == mock_x_hat