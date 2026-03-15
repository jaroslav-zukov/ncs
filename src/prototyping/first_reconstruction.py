import numpy as np

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs


def main():
    print("Testing reconstruction pipeline")

    tree_sparsity = 60
    sparse_coeff = generate_tree_sparse_coeffs(
        power=10, count=1, tree_sparsity=tree_sparsity, wavelet="haar"
    )[0]

    x_hat = measure_and_reconstruct(
        measurement_mode="gaussian",
        m=300,
        reconstruction_mode="CoSaMP",
        coeffs_x=sparse_coeff,
        target_tree_sparsity=tree_sparsity,
    )

    flat_sparse_coeffs = sparse_coeff.flat_coeffs
    flat_reconstructed_coeffs = x_hat.flat_coeffs

    difference = flat_sparse_coeffs - flat_reconstructed_coeffs
    nonzero_indices = np.nonzero(difference)[0]
    print(f"Number of wrong indices: {len(nonzero_indices)}")

    original_support = sparse_coeff.support
    reconstructed_support = x_hat.support
    print(
        f"Localized support: {len(original_support & reconstructed_support)}, "
        f"missed support({len(original_support - reconstructed_support)}): {original_support - reconstructed_support}"
    )

    for index, num in [(i, difference[i]) for i in nonzero_indices]:
        print(index, num)


if __name__ == "__main__":
    main()
