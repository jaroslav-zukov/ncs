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

    solve_on_support = None
    phi_matrix = None
    if len(compressive_sensing_operators) == 4:
        phi, phi_transpose, phi_pseudoinverse, fourth = compressive_sensing_operators
        if callable(fourth):
            solve_on_support = fourth
        else:
            phi_matrix = fourth
    else:
        phi, phi_transpose, phi_pseudoinverse = compressive_sensing_operators

    if solve_on_support is None and phi_matrix is None:
        try:
            n = x_init.n
            m = len(y)
            phi_matrix = np.zeros((m, n), dtype=np.result_type(y, float))
            for idx in range(n):
                basis_vector = np.zeros(n)
                basis_vector[idx] = 1.0
                basis_coeffs = WtCoeffs.from_flat_coeffs(
                    flat_coeffs=basis_vector,
                    root_count=x_init.root_count,
                    max_level=x_init.max_level,
                    wavelet=x_init.wavelet,
                )
                phi_matrix[:, idx] = phi(basis_coeffs)
        except Exception:
            phi_matrix = None

    def least_squares_on_support(support: set[int]) -> WtCoeffs:
        flat_coeffs = np.zeros(x_init.n)
        support_idx = np.array(sorted(support), dtype=int)
        if support_idx.size > 0:
            a_support = phi_matrix[:, support_idx]
            coeffs_support, _, _, _ = np.linalg.lstsq(a_support, y, rcond=None)
            flat_coeffs[support_idx] = coeffs_support
        return WtCoeffs.from_flat_coeffs(
            flat_coeffs=flat_coeffs,
            root_count=x_init.root_count,
            max_level=x_init.max_level,
            wavelet=x_init.wavelet,
        )

    for _ in tqdm(range(iteration_threshold), desc="CoSaMP iterations", disable=True):
        e = phi_transpose(r)
        omega_e_double_support = tree_projection(e, 2 * tree_sparsity).support
        t = x_hat.support | omega_e_double_support
        if solve_on_support is not None:
            b_coeffs_estimate = solve_on_support(y, t)
        elif phi_matrix is not None:
            b_coeffs_estimate = least_squares_on_support(t)
        else:
            b_coeffs_estimate = phi_pseudoinverse(y).on_support(t)
        x_hat = tree_projection(b_coeffs_estimate, tree_sparsity)
        r = y - phi(x_hat)

    if solve_on_support is not None:
        x_hat = solve_on_support(y, x_hat.support)
    elif phi_matrix is not None:
        x_hat = least_squares_on_support(x_hat.support)

    return x_hat


RECONSTRUCTION_ALGORITHMS = {
    "CoSaMP": cosamp_reconstruct,
}


def reconstruct(
    reconstruction_mode: str,
    y: np.ndarray,
    x_init: WtCoeffs,
    tree_sparsity: int,
    compressive_sensing_operators=None,
    measurement_op=None,
    adjoint_op=None,
    pseudo_inverse_op=None,
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

    if compressive_sensing_operators is not None:
        if any(op is None for op in compressive_sensing_operators):
            return x_init
        x_hat = reconstruction_algorithm(
            y, tree_sparsity, x_init, compressive_sensing_operators
        )
        return x_hat

    if measurement_op is None or adjoint_op is None:
        return x_init

    if pseudo_inverse_op is None:
        x_hat = reconstruction_algorithm(
            y, tree_sparsity, x_init, measurement_op, adjoint_op
        )
        return x_hat

    x_hat = reconstruction_algorithm(
        y,
        tree_sparsity,
        x_init,
        (measurement_op, adjoint_op, pseudo_inverse_op),
    )

    return x_hat
