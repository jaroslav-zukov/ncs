import numpy as np

from src.ncs.compressed_sensing_module import measure_and_reconstruct
from src.ncs.sparse_signal_generator import generate_tree_sparse_coeffs


def main():
    print("Testing reconstruction pipeline")

    tree_sparsity = int((2**12)/10)
    sparse_coeff = generate_tree_sparse_coeffs(
        power=13,
        count=1,
        tree_sparsity=tree_sparsity,
        wavelet='haar'
    )[0]

    x_hat = measure_and_reconstruct(
        measurement_mode='subsampling',
        m = 2**9,
        reconstruction_mode='CoSaMP',
        coeffs_x=sparse_coeff,
        target_tree_sparsity=tree_sparsity
    )

    flat_sparse_coeffs = sparse_coeff.flat_coeffs
    flat_reconstructed_coeffs = x_hat.flat_coeffs

    difference = flat_sparse_coeffs - flat_reconstructed_coeffs
    nonzero_indices = np.nonzero(difference)[0]
    print(f"Number of wrong indices: {len(nonzero_indices)}")
    for num in difference[nonzero_indices]:
        print(num)


if __name__ == "__main__":
    main()