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
import pywt
from matplotlib.figure import Figure

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.reconstruction_module import reconstruct
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

    Note:
        This interpretation assumes an energy-preserving sensing operator
        (E[||Φx||²] = ||x||²). For operators with different scaling, μ may be
        numerically biased and should be interpreted with care.

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
    seed: int = 0,
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
        seed: Random seed for reproducible Monte-Carlo trials.

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
            seed=seed,
        )
    else:
        template = forward_transform(np.zeros(n), wavelet)
        rng = np.random.default_rng(seed)
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


def local_coherence_matrix(
    G: np.ndarray,
    n: int,
    wavelet: str,
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    """
    Compute local coherence matrix across row bands and wavelet scales.

    Columns are partitioned by wavelet scale using the dyadic structure of
    WtCoeffs. Rows are split into r equal-sized bands with r=max_level+1.
    For each pair (band k, scale l), this computes:

        μ_{N,M}(k,l) = sqrt(
            max_{i in band_k, j in scale_l} |G_ij|^2 *
            max_{i in band_k, j in all} |G_ij|^2
        )

    Args:
        G: Effective sensing matrix Φ·Ψ⁻¹ of shape (m, n).
        n: Signal length in coefficient domain.
        wavelet: Wavelet name used to infer the dyadic coefficient layout.

    Returns:
        tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
        (local coherence matrix of shape (r, max_level+1), metadata), where
        metadata contains "band_boundaries" and "scale_boundaries" as
        half-open index intervals (start, end).

    Raises:
        ValueError: If G shape is incompatible with n.
    """
    if G.ndim != 2:
        raise ValueError("G must be a 2-D array")
    if G.shape[1] != n:
        raise ValueError("G must have n columns")

    template = forward_transform(np.zeros(n), wavelet)
    r = template.max_level + 1

    scale_boundaries: list[tuple[int, int]] = []
    scale_start = 0
    for coeff_group in template.coeff_groups:
        scale_end = scale_start + len(coeff_group)
        scale_boundaries.append((scale_start, scale_end))
        scale_start = scale_end

    m = G.shape[0]
    row_edges = np.linspace(0, m, r + 1, dtype=int)
    band_boundaries = [(int(row_edges[idx]), int(row_edges[idx + 1])) for idx in range(r)]

    abs_sq = np.abs(G) ** 2
    local_mu = np.zeros((r, r), dtype=float)

    for band_idx, (row_start, row_end) in enumerate(band_boundaries):
        band_block = abs_sq[row_start:row_end, :]
        if band_block.size == 0:
            continue
        band_max_all = float(np.max(band_block))

        for scale_idx, (col_start, col_end) in enumerate(scale_boundaries):
            local_block = abs_sq[row_start:row_end, col_start:col_end]
            local_max = float(np.max(local_block)) if local_block.size > 0 else 0.0
            local_mu[band_idx, scale_idx] = float(np.sqrt(local_max * band_max_all))

    metadata = {
        "band_boundaries": band_boundaries,
        "scale_boundaries": scale_boundaries,
    }
    return local_mu, metadata


def optimal_multilevel_allocation(
    local_sparsities: np.ndarray,
    n: int,
    wavelet: str,
    total_m: int | None,
) -> np.ndarray:
    """
    Compute multilevel measurement allocation from local sparsities.

    Uses a wavelet-regularity weighted allocation motivated by the local
    coherence framework in Adcock et al. Wavelet smoothness proxies are taken
    from PyWavelets metadata:
      - alpha proxy from decomposition filter length (dec_len),
      - nu from vanishing_moments_psi.

    Args:
        local_sparsities: Non-negative vector s_l with one entry per scale.
        n: Signal length (used to infer number of wavelet scales).
        wavelet: Wavelet name.
        total_m: Optional total measurement budget. If provided, allocation is
            rounded to integer counts that sum exactly to total_m.

    Returns:
        np.ndarray: Allocation vector m_k over scales/bands.

    Raises:
        ValueError: If local sparsity vector length mismatches scale count.
    """
    template = forward_transform(np.zeros(n), wavelet)
    level_count = template.max_level + 1

    local_sparsities = np.asarray(local_sparsities, dtype=float)
    if local_sparsities.shape != (level_count,):
        raise ValueError(
            f"local_sparsities must have shape ({level_count},), got {local_sparsities.shape}"
        )
    if np.any(local_sparsities < 0):
        raise ValueError("local_sparsities must be non-negative")

    wavelet_obj = pywt.Wavelet(wavelet)
    alpha = max((wavelet_obj.dec_len - 1) / 2.0, 1.0)
    nu = float(wavelet_obj.vanishing_moments_psi or 1)

    levels = np.arange(level_count, dtype=float)
    decay = 2.0 ** (-np.minimum(alpha, nu) * levels)
    weights = np.sqrt(local_sparsities + 1e-12) * decay

    if np.all(weights == 0):
        allocation = np.ones(level_count, dtype=float) / level_count
    else:
        allocation = weights / np.sum(weights)

    if total_m is None:
        return allocation

    if total_m < 0:
        raise ValueError("total_m must be non-negative")

    scaled = allocation * total_m
    integer_alloc = np.floor(scaled).astype(int)
    remainder = int(total_m - np.sum(integer_alloc))
    if remainder > 0:
        order = np.argsort(-(scaled - integer_alloc))
        integer_alloc[order[:remainder]] += 1

    return integer_alloc


def _reconstruct_with_custom_operator(
    coeffs_x: WtCoeffs,
    tree_sparsity: int,
    measure: MeasureFn,
    adjoint: AdjointFn,
    pseudo_inverse: PseudoInverseFn,
) -> WtCoeffs:
    y = measure(inverse_transform(coeffs_x))

    def phi(wt_coeffs: WtCoeffs) -> np.ndarray:
        return measure(inverse_transform(wt_coeffs))

    def phi_transpose(meas: np.ndarray) -> WtCoeffs:
        return forward_transform(adjoint(meas), coeffs_x.wavelet)

    def phi_pseudoinverse(meas: np.ndarray) -> WtCoeffs:
        return forward_transform(pseudo_inverse(meas), coeffs_x.wavelet)

    x_init = WtCoeffs.from_flat_coeffs(
        flat_coeffs=np.zeros(coeffs_x.n),
        root_count=coeffs_x.root_count,
        max_level=coeffs_x.max_level,
        wavelet=coeffs_x.wavelet,
    )

    return reconstruct(
        reconstruction_mode="CoSaMP",
        y=y,
        x_init=x_init,
        tree_sparsity=tree_sparsity,
        compressive_sensing_operators=(phi, phi_transpose, phi_pseudoinverse),
    )


def flip_test(
    measure_op: str | MeasureFn | MeasurementOperatorPair | MeasurementOperatorTriple,
    n: int,
    wavelet: str,
    k: int,
    n_trials: int = 50,
) -> dict[str, float]:
    """
    Run the flip test comparing structured vs coefficient-permuted recovery.

    For each trial:
      1) Generate tree-k-sparse coefficients x.
      2) Reconstruct from y = Φ·IDWT(x).
      3) Create x_flip by random permutation of coefficient indices.
      4) Reconstruct from y_flip = Φ·IDWT(x_flip).

    The returned ratio = mse_flipped / mse_structured quantifies whether the
    sensing/reconstruction pipeline benefits from wavelet-tree structure.

    Args:
        measure_op: Either a measurement mode string (registered in
            measurement_module) or operator callable/tuple.
        n: Signal length (must be power-of-two).
        wavelet: Wavelet name.
        k: Tree sparsity level.
        n_trials: Number of trials.

    Returns:
        dict[str, float]: Keys {mse_structured, mse_flipped, ratio}.
    """
    if not _is_power_of_two(n):
        raise ValueError("n must be a power of 2")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    rng = np.random.default_rng(0)
    power = int(np.log2(n))

    structured_errors: list[float] = []
    flipped_errors: list[float] = []

    for _ in range(n_trials):
        signal_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        coeffs_x = generate_tree_sparse_coeffs(
            power=power,
            count=1,
            tree_sparsity=k,
            wavelet=wavelet,
            seed=signal_seed,
        )[0]

        perm = rng.permutation(n)
        flipped_flat = coeffs_x.flat_coeffs[perm]
        coeffs_flip = WtCoeffs.from_flat_coeffs(
            flat_coeffs=flipped_flat,
            root_count=coeffs_x.root_count,
            max_level=coeffs_x.max_level,
            wavelet=wavelet,
        )

        if isinstance(measure_op, str):
            m = n // 2
            operator_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            recon_structured = measure_and_reconstruct(
                measurement_mode=measure_op,
                m=m,
                reconstruction_mode="CoSaMP",
                coeffs_x=coeffs_x,
                target_tree_sparsity=k,
                seed=operator_seed,
            )
            recon_flipped = measure_and_reconstruct(
                measurement_mode=measure_op,
                m=m,
                reconstruction_mode="CoSaMP",
                coeffs_x=coeffs_flip,
                target_tree_sparsity=k,
                seed=operator_seed,
            )
        else:
            if callable(measure_op):
                raise ValueError(
                    "flip_test requires measurement mode string or operator tuple with adjoint"
                )
            if len(measure_op) == 2:
                measure, adjoint = measure_op
                pseudo_inverse = adjoint
            elif len(measure_op) == 3:
                measure, adjoint, pseudo_inverse = measure_op
            else:
                raise ValueError("measure_op tuple must have length 2 or 3")

            recon_structured = _reconstruct_with_custom_operator(
                coeffs_x=coeffs_x,
                tree_sparsity=k,
                measure=measure,
                adjoint=adjoint,
                pseudo_inverse=pseudo_inverse,
            )
            recon_flipped = _reconstruct_with_custom_operator(
                coeffs_x=coeffs_flip,
                tree_sparsity=k,
                measure=measure,
                adjoint=adjoint,
                pseudo_inverse=pseudo_inverse,
            )

        mse_structured = float(
            np.mean((inverse_transform(coeffs_x) - inverse_transform(recon_structured)) ** 2)
        )
        mse_flipped = float(
            np.mean((inverse_transform(coeffs_flip) - inverse_transform(recon_flipped)) ** 2)
        )

        structured_errors.append(mse_structured)
        flipped_errors.append(mse_flipped)

    mean_structured = float(np.mean(structured_errors))
    mean_flipped = float(np.mean(flipped_errors))
    ratio = float(np.inf if mean_structured == 0 else mean_flipped / mean_structured)

    return {
        "mse_structured": mean_structured,
        "mse_flipped": mean_flipped,
        "ratio": ratio,
    }


def phase_transition_grid(
    measurement_mode: str,
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
        measurement_mode: Measurement mode name supported by
            measurement_module.MEASUREMENT_OPERATORS.
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
                    measurement_mode=measurement_mode,
                    m=int(m),
                    reconstruction_mode="CoSaMP",
                    coeffs_x=coeffs_x,
                    target_tree_sparsity=int(k),
                    seed=operator_seed,
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
