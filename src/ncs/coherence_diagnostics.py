"""
Incoherence diagnostics for measurement operators in wavelet CS.

This module implements practical validation tools for the effective sensing
operator A = Φ·Ψ⁻¹, where Φ is a measurement operator and Ψ⁻¹ is wavelet
synthesis (inverse DWT). The functions support:

- Gram matrix materialisation of A by probing wavelet-basis atoms,
- mutual coherence estimation,
- heatmap visualisation of coefficient/measurement interactions,
- empirical RIP-constant estimation,
- phase-transition style recovery-probability sweeps.

The expected measurement operator format matches measurement_module factory
outputs: (measure, adjoint, pseudo_inverse).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs
from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs

MeasureFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
AdjointFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
PseudoInverseFn: TypeAlias = Callable[[np.ndarray], np.ndarray]
MeasurementOperatorTriple: TypeAlias = tuple[MeasureFn, AdjointFn, PseudoInverseFn]
MeasurementOperatorPair: TypeAlias = tuple[MeasureFn, AdjointFn]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resolve_measure_fn(
    measure_op: MeasureFn | MeasurementOperatorTriple | MeasurementOperatorPair,
) -> MeasureFn:
    if callable(measure_op):
        return measure_op

    if len(measure_op) not in {2, 3}:
        raise ValueError("Measurement operator must be callable or a 2/3-tuple")

    measure_fn = measure_op[0]
    return measure_fn


def _dyadic_level_boundaries(n: int) -> list[int]:
    if not _is_power_of_two(n):
        return []

    max_level = int(np.log2(n))
    level_sizes = [1]
    if max_level >= 1:
        level_sizes.append(1)
    for level in range(2, max_level + 1):
        level_sizes.append(2 ** (level - 1))

    boundaries: list[int] = []
    running = 0
    for level_size in level_sizes[:-1]:
        running += level_size
        boundaries.append(running)

    return boundaries


def compute_gram_matrix(
    measure_op: MeasureFn | MeasurementOperatorTriple,
    n: int,
    wavelet: str,
) -> np.ndarray:
    """
    Materialise the effective sensing matrix G = Φ·Ψ⁻¹.

    Builds wavelet-basis vectors e_j in coefficient coordinates and computes
    the j-th column as G[:, j] = Φ(Ψ⁻¹ e_j).

    Args:
        measure_op: Measurement operator callable measure(x) or full operator
            triple (measure, adjoint, pseudo_inverse).
        n: Signal / coefficient length.
        wavelet: Orthogonal wavelet name for inverse_transform synthesis.

    Returns:
        np.ndarray: Effective Gram-like sensing matrix of shape (m, n), where
        m is inferred from the measurement output length.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    measure_fn = _resolve_measure_fn(measure_op)
    template_coeffs = forward_transform(np.zeros(n), wavelet)

    first_basis = np.zeros(n)
    first_basis[0] = 1.0
    first_coeffs = WtCoeffs.from_flat_coeffs(
        flat_coeffs=first_basis,
        root_count=template_coeffs.root_count,
        max_level=template_coeffs.max_level,
        wavelet=wavelet,
    )
    first_column = np.asarray(measure_fn(inverse_transform(first_coeffs)))

    matrix = np.zeros((len(first_column), n), dtype=np.result_type(first_column.dtype, float))
    matrix[:, 0] = first_column

    for index in range(1, n):
        basis = np.zeros(n)
        basis[index] = 1.0
        coeffs = WtCoeffs.from_flat_coeffs(
            flat_coeffs=basis,
            root_count=template_coeffs.root_count,
            max_level=template_coeffs.max_level,
            wavelet=wavelet,
        )
        matrix[:, index] = measure_fn(inverse_transform(coeffs))

    return matrix


def mutual_coherence(G: np.ndarray) -> float:
    """
    Compute mutual coherence proxy for a sensing matrix.

    The metric is:
        μ = √n * max_{i,j} |G_norm[i, j]|
    where each column of G_norm is unit-norm.

    Args:
        G: Matrix of shape (m, n).

    Returns:
        float: Coherence value μ.

    Raises:
        ValueError: If G is not 2-D or contains a zero-norm column.
    """
    if G.ndim != 2:
        raise ValueError("G must be a 2-D array")

    _, n = G.shape
    column_norms = np.linalg.norm(G, axis=0)
    if np.any(column_norms == 0):
        raise ValueError("G contains at least one zero-norm column")

    normalized = G / column_norms
    return float(np.sqrt(n) * np.max(np.abs(normalized)))


def coherence_heatmap(G: np.ndarray) -> Figure:
    """
    Plot |G_ij|^2 heatmap for incoherence diagnostics.

    Args:
        G: Matrix of shape (m, n) representing Φ·Ψ⁻¹.

    Returns:
        matplotlib.figure.Figure: Figure containing the heatmap.
    """
    _, n = G.shape

    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(np.abs(G) ** 2, origin="lower", aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label=r"$|G_{ij}|^2$")

    for boundary in _dyadic_level_boundaries(n):
        ax.axvline(boundary - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.9)

    ax.set_xlabel("Wavelet coefficient index")
    ax.set_ylabel("Measurement index")
    ax.set_title(r"Coherence heatmap of $|\Phi \Psi^{-1}|^2$")
    fig.tight_layout()

    return fig


def empirical_rip_constant(
    measure_op: MeasureFn | MeasurementOperatorTriple,
    n: int,
    wavelet: str,
    k: int,
    n_trials: int = 1000,
    tree_sparse: bool = True,
) -> dict[str, float]:
    """
    Estimate empirical RIP distortion for k-sparse inputs.

    For each random sparse coefficient vector x, this function computes
    ratio r = ||ΦΨ⁻¹x||² / ||x||² and aggregates summary statistics.

    Args:
        measure_op: Measurement operator callable or operator triple.
        n: Signal length (must be power-of-two for wavelet-tree generator).
        wavelet: Orthogonal wavelet name.
        k: Sparsity level.
        n_trials: Number of Monte-Carlo trials.
        tree_sparse: If True, sample tree-k-sparse vectors via
            generate_tree_sparse_coeffs; else use uniform random k-support.

    Returns:
        dict[str, float]: Summary containing keys
        {delta_k, mean_ratio, std_ratio, min_ratio, max_ratio}.

    Raises:
        ValueError: If n is not power-of-two, k is invalid, or n_trials < 1.
    """
    if not _is_power_of_two(n):
        raise ValueError("n must be a power of 2")
    if not (1 <= k <= n):
        raise ValueError("k must satisfy 1 <= k <= n")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    measure_fn = _resolve_measure_fn(measure_op)
    ratios = np.zeros(n_trials, dtype=float)

    if tree_sparse:
        power = int(np.log2(n))
        sparse_coeffs = generate_tree_sparse_coeffs(
            power=power,
            count=n_trials,
            tree_sparsity=k,
            wavelet=wavelet,
            seed=0,
        )
    else:
        template = forward_transform(np.zeros(n), wavelet)
        rng = np.random.default_rng(0)
        sparse_coeffs = []
        for _ in range(n_trials):
            support = rng.choice(n, size=k, replace=False)
            flat = np.zeros(n)
            flat[support] = rng.standard_normal(k)
            sparse_coeffs.append(
                WtCoeffs.from_flat_coeffs(
                    flat_coeffs=flat,
                    root_count=template.root_count,
                    max_level=template.max_level,
                    wavelet=wavelet,
                )
            )

    for index, coeffs in enumerate(sparse_coeffs):
        measurement = measure_fn(inverse_transform(coeffs))
        numerator = np.sum(np.abs(measurement) ** 2)
        denominator = np.sum(np.abs(coeffs.flat_coeffs) ** 2)
        ratios[index] = float(numerator / denominator)

    return {
        "delta_k": float(np.max(np.abs(ratios - 1.0))),
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
    }


def phase_transition_grid(
    measure_op_factory: Callable[[int, int, int | None], MeasurementOperatorTriple],
    n: int,
    wavelet: str,
    m_values: Iterable[int],
    k_values: Iterable[int],
    n_trials: int = 50,
) -> pd.DataFrame:
    """
    Estimate recovery probability grid over measurement and sparsity levels.

    For each pair (m, k), runs n_trials independent recoveries and computes
    success probability under threshold MSE < 1e-6.

    Args:
        measure_op_factory: Callable with signature (n, m, seed) ->
            (measure, adjoint, pseudo_inverse).
        n: Signal length (power-of-two required).
        wavelet: Wavelet name used for sparse generation and reconstruction.
        m_values: Measurement counts to evaluate.
        k_values: Sparsity levels to evaluate.
        n_trials: Number of trials per (m, k) pair.

    Returns:
        pd.DataFrame: Columns [m, k, recovery_probability].
    """
    if not _is_power_of_two(n):
        raise ValueError("n must be a power of 2")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    power = int(np.log2(n))
    rng = np.random.default_rng(0)

    rows: list[dict[str, Any]] = []

    for m in m_values:
        for k in k_values:
            successes = 0
            for _ in range(n_trials):
                signal_seed = int(rng.integers(0, np.iinfo(np.int32).max))
                operator_seed = int(rng.integers(0, np.iinfo(np.int32).max))

                coeffs_x = generate_tree_sparse_coeffs(
                    power=power,
                    count=1,
                    tree_sparsity=int(k),
                    wavelet=wavelet,
                    seed=signal_seed,
                )[0]

                reconstructed = measure_and_reconstruct(
                    measurement_mode="subsampling",
                    m=int(m),
                    reconstruction_mode="CoSaMP",
                    coeffs_x=coeffs_x,
                    target_tree_sparsity=int(k),
                    seed=operator_seed,
                    measurement_op_factory=measure_op_factory,
                )

                mse = np.mean(
                    (inverse_transform(coeffs_x) - inverse_transform(reconstructed)) ** 2
                )
                successes += int(mse < 1e-6)

            rows.append(
                {
                    "m": int(m),
                    "k": int(k),
                    "recovery_probability": float(successes / n_trials),
                }
            )

    return pd.DataFrame(rows, columns=["m", "k", "recovery_probability"])
