import numpy as np

from ncs import load_signal
from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import get_orthogonal_wavelets, forward_transform, inverse_transform

signal = load_signal(power=13, count=1)[0]


coeffs = {}
results = {}
for wavelet in get_orthogonal_wavelets():
    coeffs[wavelet] = forward_transform(signal, wavelet)
    results[wavelet] = []

    for tree_sparsity in np.linspace(80, 1600, 10):
        tree_sparse_coeff = tree_projection(coeffs[wavelet], tree_sparsity)
        tree_sparse_signal = inverse_transform(tree_sparse_coeff)
        mse = np.mean((signal - tree_sparse_signal) ** 2)
        results[wavelet].append(
            {"tree_sparsity": tree_sparsity, "mse": mse}
        )