import numpy as np

from src.ncs.exact_tree_projection import tree_projection
from src.ncs.wavelet_module import forward_transform, inverse_transform
from src.ncs.wt_coeffs import WtCoeffs


def generate_tree_sparse_coeffs(power: int, count: int, tree_sparsity: int, wavelet: str) -> list[WtCoeffs]:
    n = 2 ** power

    seed = None
    np.random.seed(seed)
    random_signals = [np.random.randint(low=-300, high=300, size=n) for _ in range(count)]

    random_wt_coeffs: list[WtCoeffs] = [forward_transform(signal, wavelet) for signal in random_signals]
    tree_sparse_coeffs: list[WtCoeffs] = [tree_projection(wt_coeffs, tree_sparsity) for wt_coeffs in random_wt_coeffs]

    return tree_sparse_coeffs


def generate_tree_sparse_signals(power: int, count: int, tree_sparsity: int, wavelet: str):
    tree_sparse_coeffs: list[WtCoeffs] = generate_tree_sparse_coeffs(power, count, tree_sparsity, wavelet)
    tree_sparse_signals: list[np.ndarray] = [inverse_transform(wt_coeffs) for wt_coeffs in tree_sparse_coeffs]

    return tree_sparse_signals
