import numpy as np

from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs


def generate_tree_sparse_coeffs(
    power: int,
    count: int,
    tree_sparsity: int,
    wavelet: str,
    seed: int = None,
) -> list[WtCoeffs]:
    np.random.seed(seed)
    random_signals = [
        np.random.randint(low=-300, high=300, size=2**power) for _ in range(count)
    ]

    random_wt_coeffs: list[WtCoeffs] = [
        forward_transform(signal, wavelet) for signal in random_signals
    ]
    tree_sparse_coeffs: list[WtCoeffs] = [
        tree_projection(wt_coeffs, tree_sparsity) for wt_coeffs in random_wt_coeffs
    ]

    return tree_sparse_coeffs


def generate_tree_sparse_signals(
    power: int,
    count: int,
    tree_sparsity: int,
    wavelet: str,
    seed: int = None,
):
    tree_sparse_coeffs: list[WtCoeffs] = generate_tree_sparse_coeffs(
        power, count, tree_sparsity, wavelet, seed
    )
    tree_sparse_signals: list[np.ndarray] = [
        inverse_transform(wt_coeffs) for wt_coeffs in tree_sparse_coeffs
    ]

    return tree_sparse_signals


def add_noise_to_coeffs(
    tree_sparse_signals: list[WtCoeffs],
    noise_epsilon: float,
    noise_mode: str,
    seed: int | None = None,
):
    supported_noise_modes: set[str] = {"gaussian", "uniform"}
    if noise_mode not in supported_noise_modes:
        raise ValueError(f"Noise mode {noise_mode} not supported. (use {supported_noise_modes})")

    if seed is not None:
        np.random.seed(seed)

    noisy_coeffs: list[WtCoeffs] = []

    for wt_coeffs in tree_sparse_signals:
        signal_z = inverse_transform(wt_coeffs)

        noise_diameter = int(noise_epsilon * 600)

        if noise_mode == "gaussian":
            noise = np.random.randn(wt_coeffs.n) * noise_diameter / 3
            noisy_signal = signal_z + noise
        else:
            noise = np.random.randint(
                low=-noise_diameter, high=noise_diameter, size=wt_coeffs.n
            )
            noisy_signal = signal_z + noise
        noisy_coeffs.append(forward_transform(noisy_signal, wt_coeffs.wavelet))

    return noisy_coeffs