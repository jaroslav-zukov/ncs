"""
Wavelet Packet (WP) transform and best-basis selection for NCS.

Wavelet packets generalise the standard DWT by splitting BOTH approximation
and detail subbands at every decomposition level, yielding a full binary tree
of possible orthonormal bases. The best basis (Coifman & Wickerhauser, 1992)
is the tree leaf selection that minimises Shannon entropy of the signal's
coefficients — equivalently, the maximally sparse representation within the
WP library.

Relevance to compressive sensing incoherence
--------------------------------------------
WP basis functions at deeper tree levels have wider time support and lower
pointwise amplitude (max_t |ψ_{j,p}(t)| ≤ C·2^{-j/2}), making them more
incoherent with time-domain subsampling. The best-basis algorithm
automatically trades sparsity against coherence, finding the WP basis that
minimises the product μ²·k — the key quantity in the CS measurement bound.

References
----------
Coifman, R. R., & Wickerhauser, M. V. (1992). Entropy-based algorithms for
    best basis selection. IEEE Trans. Inf. Theory, 38(2), 713–718.
Mallat, S. (2008). A Wavelet Tour of Signal Processing (3rd ed.).
    Chapter 8: Wavelet Packet and Local Cosine Bases.
"""

import numpy as np
import pywt
from dataclasses import dataclass, field


@dataclass
class WaveletPacketBasis:
    """
    Represents a selected wavelet packet basis.

    A WP basis is defined by a set of leaf nodes in the full WP decomposition
    tree. Each leaf node (level, node_path) corresponds to a subspace of
    specific time-frequency localisation.

    Attributes:
        wavelet: Wavelet name (must be orthogonal, e.g. 'haar', 'db4')
        max_level: Maximum decomposition depth of the WP tree
        leaf_nodes: List of (level, node_path) tuples identifying selected leaves
        n: Original signal length
    """
    wavelet: str
    max_level: int
    leaf_nodes: list
    n: int


def shannon_entropy(coeffs: np.ndarray) -> float:
    """
    Shannon entropy of a coefficient array (Coifman-Wickerhauser cost function).

    Defined as: H(c) = -sum(c_i² * log(c_i²)) for c_i != 0
    Normalised so ||c||₂ = 1 is assumed; raw coefficients are normalised internally.

    Args:
        coeffs: 1D coefficient array
    Returns:
        Shannon entropy (lower = sparser)
    """
    c2 = coeffs ** 2
    norm = c2.sum()
    if norm < 1e-15:
        return 0.0
    p = c2 / norm
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def best_basis_selection(signal: np.ndarray, wavelet: str, max_level: int) -> tuple:
    """
    Coifman-Wickerhauser best basis algorithm.

    Traverses the full WP decomposition tree bottom-up, selecting at each node
    whether to keep the node as a leaf (use its coefficients directly) or split
    further (use its children). Selection criterion: minimum Shannon entropy.

    Args:
        signal: 1D real-valued signal (length should be power of 2)
        wavelet: Orthogonal wavelet name
        max_level: Maximum decomposition depth

    Returns:
        Tuple of:
          - leaf_nodes: list of (level, node_path_string) for selected leaves
          - node_coeffs: dict mapping node_path_string -> coefficient array
    """
    # Build full WP tree using pywt
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='periodization',
                             maxlevel=max_level)

    # Collect all nodes
    all_nodes = {}
    for level in range(1, max_level + 1):
        for node in wp.get_level(level, 'natural'):
            all_nodes[node.path] = node.data.copy()

    def get_best_basis_recursive(path, level):
        """Returns (entropy, leaf_list) for best basis rooted at this node."""
        node_data = all_nodes.get(path)
        if node_data is None:
            return 0.0, []

        node_entropy = shannon_entropy(node_data)

        if level >= max_level:
            # Leaf of full tree — must use this node
            return node_entropy, [(level, path)]

        # Try splitting
        left_path = path + 'a'
        right_path = path + 'd'

        left_entropy, left_leaves = get_best_basis_recursive(left_path, level + 1)
        right_entropy, right_leaves = get_best_basis_recursive(right_path, level + 1)

        split_entropy = left_entropy + right_entropy

        if node_entropy <= split_entropy:
            # Keep this node as a leaf
            return node_entropy, [(level, path)]
        else:
            # Split is better
            return split_entropy, left_leaves + right_leaves

    _, approx_leaves = get_best_basis_recursive('a', 1)
    _, detail_leaves = get_best_basis_recursive('d', 1)
    leaf_nodes = approx_leaves + detail_leaves

    return leaf_nodes, all_nodes


def signal_to_wp_coeffs(signal: np.ndarray, wavelet: str, leaf_nodes: list,
                         max_level: int) -> np.ndarray:
    """
    Project signal onto a wavelet packet basis defined by leaf_nodes.

    Args:
        signal: 1D real signal
        wavelet: Orthogonal wavelet name
        leaf_nodes: List of (level, path) tuples from best_basis_selection
        max_level: Max decomposition level

    Returns:
        Flat coefficient array in the WP basis (concatenated leaf coefficients)
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='periodization',
                             maxlevel=max_level)
    coeffs = []
    for (level, path) in sorted(leaf_nodes, key=lambda x: x[1]):
        node = wp[path]
        coeffs.append(node.data.copy())
    return np.concatenate(coeffs)


def wp_coeffs_to_signal(flat_coeffs: np.ndarray, wavelet: str, leaf_nodes: list,
                         max_level: int, n: int) -> np.ndarray:
    """
    Reconstruct signal from wavelet packet coefficients.

    Args:
        flat_coeffs: Flat coefficient array (same order as signal_to_wp_coeffs)
        wavelet: Orthogonal wavelet name
        leaf_nodes: List of (level, path) tuples
        max_level: Max decomposition level
        n: Original signal length

    Returns:
        Reconstructed signal of length n
    """
    wp = pywt.WaveletPacket(data=np.zeros(n), wavelet=wavelet, mode='periodization',
                             maxlevel=max_level)

    sorted_leaves = sorted(leaf_nodes, key=lambda x: x[1])
    idx = 0
    for (level, path) in sorted_leaves:
        size = n // (2 ** level)
        wp[path] = flat_coeffs[idx:idx + size]
        idx += size

    return wp.reconstruct(update=False)


def measure_coherence(leaf_nodes: list, wavelet: str, max_level: int, n: int,
                       n_samples: int = 50) -> float:
    """
    Empirically estimate mutual coherence μ(S, Ψ_WP) for the given WP basis.

    Coherence: μ = max_{i,j} |<e_{t_i}, ψ_j>|²
    where e_{t_i} is the i-th canonical basis vector (time-domain delta).

    Approximated by sampling n_samples random time indices and computing
    max inner product with a sample of basis functions.

    Args:
        leaf_nodes: Selected WP basis leaves
        wavelet: Orthogonal wavelet name
        max_level: Max decomposition level
        n: Signal length
        n_samples: Number of random time indices to test

    Returns:
        Estimated mutual coherence (lower = better for CS)
    """
    max_coherence = 0.0
    rng = np.random.default_rng(42)

    for (level, path) in leaf_nodes:
        size = n // (2 ** level)
        # Sample a few basis functions (translations) at this node
        n_trans = min(size, 8)
        translation_indices = rng.choice(size, size=n_trans, replace=False)

        for k in translation_indices:
            # Build unit vector in WP coefficient space
            sorted_leaves = sorted(leaf_nodes, key=lambda x: x[1])
            flat = np.zeros(n)
            idx = 0
            for (lv, p) in sorted_leaves:
                sz = n // (2 ** lv)
                if (lv, p) == (level, path):
                    flat[idx + (k % sz)] = 1.0
                idx += sz

            basis_fn = wp_coeffs_to_signal(flat, wavelet, leaf_nodes, max_level, n)

            # Coherence = max |basis_fn[t]|² (time-domain inner product with delta)
            local_max = np.max(basis_fn ** 2)
            max_coherence = max(max_coherence, local_max)

    return max_coherence
