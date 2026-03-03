"""
Signal reconstruction via CoSaMP with tree-sparse wavelet projection.

This module implements the CoSaMP (Compressive Sampling Matching Pursuit)
algorithm specialised for model-based compressive sensing where the sparsity
model is *tree-sparsity* in the wavelet coefficient domain.

CoSaMP overview
---------------
CoSaMP (Needell & Tropp, 2009) is an iterative greedy algorithm for
recovering a sparse signal x ∈ ℝⁿ from m compressed measurements
y = Φx + noise, where Φ ∈ ℝ^{m×n} (m << n) satisfies the Restricted
Isometry Property (RIP).

Each iteration refines an estimate x̂ by:
  1. Computing a proxy vector in the signal domain via the adjoint.
  2. Identifying the 2k largest (in magnitude) coordinates of the proxy.
  3. Merging the identified support with the current estimate's support.
  4. Solving a least-squares problem on the merged support.
  5. Pruning the least-squares solution back to the k-sparse model.
  6. Updating the residual r = y − Φx̂.

Model-based CoSaMP (Baraniuk et al., 2010) replaces the standard k-sparse
projections with a structured sparsity projection — here, the *exact tree
projection* of Cartis & Thompson (2013) — which enforces connected-subtree
support in the wavelet coefficient tree.  This leads to faster convergence
and better recovery with fewer measurements for piecewise-smooth signals.

References
----------
Needell, D. & Tropp, J. A. (2009). CoSaMP: Iterative signal recovery from
    incomplete and inaccurate samples. Applied and Computational Harmonic
    Analysis, 26(3), 301–321.
Baraniuk, R., Cevher, V., Duarte, M., & Hegde, C. (2010). Model-based
    compressive sensing. IEEE Transactions on Information Theory, 56(4),
    1982–2001.
Cartis, C. & Thompson, A. (2013). A reformulation of the exact tree
    projection algorithm. arXiv:1302.1720.
"""

from typing import Callable

import numpy as np
from tqdm import tqdm

from ncs.exact_tree_projection import tree_projection
from ncs.wavelet_module import forward_transform, inverse_transform
from ncs.wt_coeffs import WtCoeffs


def cosamp_reconstruct(
    y: np.ndarray,
    tree_sparsity: int,
    x_init: WtCoeffs,
    compressive_sensing_operators,
) -> WtCoeffs:
    """
    Reconstruct a tree-sparse wavelet signal from compressed measurements.

    Implements model-based CoSaMP (Needell & Tropp, 2009; Baraniuk et al.,
    2010) with the exact tree projection (Cartis & Thompson, 2013) as the
    sparsity model.  Runs for a fixed number of iterations (20).

    Algorithm per iteration
    -----------------------
    Let x̂ be the current estimate (initialised to x_init, typically zero),
    r = y − Φx̂ the current residual, and k = tree_sparsity.

    1. **Proxy** e = Φᵀr
       Apply the adjoint measurement operator to the residual to obtain a
       proxy for the true coefficient vector in the wavelet domain.

    2. **Identify 2k support** Ω = support(P_tree(e, 2k))
       Project e onto the tree-2k-sparse model to find the 2k-dimensional
       connected subtree support that best explains the proxy.

    3. **Merge** T = support(x̂) ∪ Ω
       Merge the identified support with the current estimate's support to
       form an extended candidate support.

    4. **Least squares on T**  b = Φ†(y) restricted to T
       Compute the least-squares signal estimate b on the merged support T
       by applying the pseudo-inverse of Φ and zeroing out coordinates
       outside T.

    5. **Prune to tree-k-sparse** x̂ = P_tree(b, k)
       Project b onto the tree-k-sparse model (exact tree projection) to
       enforce the structured sparsity constraint and obtain the updated
       estimate.

    6. **Update residual** r = y − Φx̂
       Recompute the residual for the next iteration.

    Args:
        y: Measurement vector of shape (m,).  y = Φx_true (+ noise).
        tree_sparsity: Target tree-sparsity level k.  The algorithm
            identifies 2k-sparse subtrees internally (step 2).
        x_init: Initial wavelet coefficient estimate (usually all zeros).
            Determines the wavelet basis, decomposition level, and root
            count used throughout.
        compressive_sensing_operators: Tuple (phi, phi_transpose,
            phi_pseudoinverse) where:
              phi(wt_coeffs) → y             forward measurement
              phi_transpose(y) → WtCoeffs    adjoint
              phi_pseudoinverse(y) → WtCoeffs  least-squares pseudo-inverse

    Returns:
        WtCoeffs: Estimated wavelet coefficient vector x̂ after 20 iterations.
    """
    x_hat = x_init
    r = np.copy(y)
    iteration_threshold = 20

    (phi, phi_transpose, phi_pseudoinverse) = compressive_sensing_operators

    for _ in tqdm(range(iteration_threshold), desc="CoSaMP iterations", disable=True):
        e = phi_transpose(r)
        omega_e_double_support = tree_projection(e, 2 * tree_sparsity).support
        t = x_hat.support | omega_e_double_support
        b_coeffs_estimate = phi_pseudoinverse(y).on_support(t)
        x_hat = tree_projection(b_coeffs_estimate, tree_sparsity)
        r = y - phi(x_hat)
    return x_hat


# TODO: Update types
# MeasurementFunction = Callable[[np.ndarray], np.ndarray]
# ReconstructionAlgorithm = Callable[
#     [np.ndarray, int, WtCoeffs, MeasurementFunction, MeasurementFunction],
#     WtCoeffs,
# ]

# RECONSTRUCTION_ALGORITHMS: dict[str, ReconstructionAlgorithm] = {
RECONSTRUCTION_ALGORITHMS = {
    "CoSaMP": cosamp_reconstruct,
}


def reconstruct(
    reconstruction_mode: str,
    y: np.ndarray,
    x_init: WtCoeffs,
    tree_sparsity: int,
    compressive_sensing_operators,
):
    """
    Dispatcher: run a named reconstruction algorithm.

    Currently supported algorithms: "CoSaMP".

    Args:
        reconstruction_mode: Algorithm name string.  Must be a key in
            RECONSTRUCTION_ALGORITHMS.
        y: Measurement vector of shape (m,).
        x_init: Initial wavelet coefficient estimate (typically zero).
        tree_sparsity: Target tree-sparsity level k passed to the algorithm.
        compressive_sensing_operators: Tuple (phi, phi_transpose,
            phi_pseudoinverse) as required by the chosen algorithm.

    Returns:
        WtCoeffs: Recovered wavelet coefficient estimate x̂.

    Raises:
        ValueError: If reconstruction_mode is not a supported algorithm name.
    """
    if reconstruction_mode not in RECONSTRUCTION_ALGORITHMS.keys():
        raise ValueError(
            f"Reconstruction mode {reconstruction_mode} not supported ({RECONSTRUCTION_ALGORITHMS.keys()})"
        )

    reconstruction_algorithm = RECONSTRUCTION_ALGORITHMS[reconstruction_mode]

    x_hat = reconstruction_algorithm(
        y, tree_sparsity, x_init, compressive_sensing_operators
    )

    return x_hat
