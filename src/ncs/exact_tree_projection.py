"""
Exact tree-sparse projection for wavelet-domain compressive sensing.

Tree-sparsity model
-------------------
A wavelet coefficient vector is *tree-k-sparse* if it has at most k non-zero
coefficients whose positions form a **connected subtree** of the wavelet
coefficient tree.  In the standard dyadic wavelet tree, each node at level j
has two children at level j+1 (detail coefficients at finer scale), and the
root(s) represent the coarsest-scale approximation coefficients.

Why tree-sparsity?
------------------
Piecewise-smooth signals (such as Oxford Nanopore ionic-current squiggles)
produce wavelet coefficients that are large only near discontinuities (event
boundaries) and at coarser scales.  These large coefficients naturally form
**connected subtrees**: a large detail coefficient at a coarse level implies
that at least some descendant detail coefficients (capturing the same spatial
region at finer resolution) are also significant.  This is sometimes called
the "persistence across scales" property.

Enforcing the connected-subtree constraint rather than plain sparsity:
  • Reduces the number of measurements needed for recovery (stronger model).
  • Eliminates isolated noisy coefficients that plain sparsity would retain.
  • Leads to faster convergence of model-based CoSaMP.

References: Baraniuk, R. (1999). Optimal tree approximation with wavelets.
    SPIE Wavelet Applications in Signal and Image Processing VII.
    Cartis, C. & Thompson, A. (2013). A reformulation of the exact tree
    projection algorithm. arXiv:1302.1720.

Dynamic-programming algorithm (Cartis & Thompson 2013)
------------------------------------------------------
Given y ∈ ℝⁿ (wavelet coefficients, possibly including an initial estimate)
and target sparsity k, the algorithm finds the connected subtree T* of size k
that maximises Σᵢ∈T* yᵢ² — i.e., the best k-term connected-subtree
approximation in ℓ₂ norm.

Two tables are maintained:
  f[(i, l)] = maximum squared energy achievable by selecting l coefficients
              from the subtree rooted at node i.
  g[(i, l)] = list of child-budget splits that achieve f[(i, l)].

The algorithm proceeds **bottom-up** from leaves to roots, filling f and g
level by level.  After the tables are complete a **traceback** pass
(top-down) recovers the optimal support set.

Complexity: O(k² · n) time, O(k · n) space.

The implementation uses a 1-indexed MathArray wrapper to match the
1-indexed notation in the Cartis & Thompson paper.
"""

from ncs.wt_coeffs import WtCoeffs


class MathArray:
    """
    1-indexed wrapper around a list, matching the paper's 1-based indexing.

    Cartis & Thompson (2013) use 1-based node numbering throughout their DP
    formulation.  This wrapper translates 1-based access (index 1 … n) to
    Python's 0-based list indices, allowing the implementation to follow the
    paper's notation directly without off-by-one translation at every site.

    Args:
        data: A Python list of any element type.

    Example:
        arr = MathArray([10, 20, 30])
        arr[1]  # → 10  (maps to data[0])
        arr[3]  # → 30  (maps to data[2])
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index - 1]

    def __setitem__(self, index, value):
        self.data[index - 1] = value

    def __len__(self):
        return len(self.data)


def tree_projection(wt_coeffs: WtCoeffs, k: int) -> WtCoeffs:
    """
    Project wavelet coefficients onto the best tree-k-sparse approximation.

    Finds the connected subtree T* of the wavelet coefficient tree with
    exactly k nodes that maximises Σᵢ∈T* wt_coeffs[i]², then returns the
    coefficient vector that retains only the coefficients on T* (zeroing all
    others).

    This implements the exact DP algorithm of Cartis & Thompson (2013) for
    the dyadic (d=2) wavelet tree with possibly multiple roots (root_count
    approximation coefficients at the coarsest scale).

    DP formulation
    --------------
    For each node i and budget l ∈ {0, …, k}:

        f[(i, l)] = max energy from l coefficients in subtree rooted at i,
                    where the root i *may* be included (with budget 1) or not.

        g[(i, l)] = [s₁, s₂]: how the budget l-1 (excluding node i itself)
                    is split between the two children of i.

    Base case (leaves):
        f[(leaf, 0)] = 0,  f[(leaf, 1)] = y[leaf]²
        g[(leaf, 0)] = g[(leaf, 1)] = [0, 0]

    Recursion (internal node i at level j with children c₁ = 2(i-1)+1,
    c₂ = 2(i-1)+2 in the standard dyadic tree):
        For each l ≥ 2 and each child r ∈ {1, 2}, find the split
        ŝ = argmax_s  f[(cᵣ, s)] + f[(i, l-s)]
        constrained to feasible budget ranges, then update f[(i, l)].

    Bottom-up pass: Leaves → internal nodes → roots → virtual root (node 0).
    Traceback pass: From node 0 downward, use g tables to recover which
    nodes are selected and propagate child budgets.

    Complexity
    ----------
    Time: O(k² · n) — for each of the n nodes, the inner budget loop is O(k²).
    Space: O(k · n) — storing f and g tables.

    Args:
        wt_coeffs: WtCoeffs object containing the wavelet coefficients to
            project.  Must have a valid dyadic tree structure (root_count,
            max_level, flat_coeffs in canonical flat layout).
        k: Target tree-sparsity level.  Must satisfy 1 ≤ k ≤ n.
            k is cast to int internally (supports passing 2*tree_sparsity
            as a float from CoSaMP).

    Returns:
        WtCoeffs: New coefficient object with the same tree structure, wavelet,
            and root_count as the input, but with all coefficients outside the
            optimal k-node connected subtree set to zero.  The support
            property of the result gives the selected indices.
    """
    root_count = wt_coeffs.root_count

    k = int(k)
    d = 2
    y = MathArray(wt_coeffs.flat_coeffs)
    n = len(y)
    max_level = wt_coeffs.max_level

    def subtree_size(level):
        return int(min((d ** (max_level + 1 - level) - 1) / (d - 1), k - level))

    f = {}
    g = {}

    f_temp = {}
    g_temp = {}

    # leaves iterator
    for i in range(
            root_count * (d ** (max_level - 1)) + 1, root_count * (d ** max_level) + 1
    ):
        f[(i, 0)] = 0
        f[(i, 1)] = y[i] ** 2

        g[(i, 0)] = [0, 0]
        g[(i, 1)] = [0, 0]

    # level iterator
    for j in range(max_level - 1, 0, -1):
        for i in range(root_count * (d ** (j - 1)) + 1, (root_count * (d ** j)) + 1):
            f[(i, 0)] = 0
            f[(i, 1)] = y[i] ** 2
            g[(i, 0)] = [0, 0]
            g[(i, 1)] = [0, 0]

            for r in range(1, d + 1):  # r just like in paper
                for l in range(
                        2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    s_minus = max(0, l - ((r - 1) * subtree_size(j + 1) + 1))
                    s_plus = min(l - 1, subtree_size(j + 1))

                    s_hat = max(
                        range(s_minus, s_plus + 1),
                        key=lambda s: f[(d * (i - 1) + r, s)] + f[(i, l - s)],
                    )
                    # d * (i - 1) + r -> refers to the r-th child of node i in a d-ary tree/forest (for level 1 and lower)
                    f_temp[(i, l)] = f[d * (i - 1) + r, s_hat] + f[(i, l - s_hat)]

                    g_temp[(i, l)] = list(g[(i, l - s_hat)])
                    # updating the r-th coordinate of g_temp
                    g_temp[(i, l)][r - 1] = s_hat

                for l in range(
                        2, min(subtree_size(j), r * subtree_size(j + 1) + 1) + 1
                ):
                    f[(i, l)] = f_temp[(i, l)]
                    g[(i, l)] = g_temp[(i, l)]

    for i in range(1, root_count + 1):
        f[(i, 0)] = 0
        f[(i, 1)] = y[i] ** 2
        g[(i, 0)] = [0, 0]
        g[(i, 1)] = [0, 0]

        for r in range(
                2, d + 1
        ):  # r=2 the only child, I don't like this setup, should be simpler!
            for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
                s_minus = max(0, l - ((r - 2) * subtree_size(1) + 1))
                s_plus = min(l - 1, subtree_size(1))
                # We calculate a child index of a root i like this i+root_count, because level 1 is same size as root level
                s_hat = max(
                    range(s_minus, s_plus + 1),
                    key=lambda s: f[(i + root_count, s)] + f[(i, l - s)],
                )

                f_temp[(i, l)] = f[(i + root_count, s_hat)] + f[(i, l - s_hat)]

                g_temp[(i, l)] = list(g[(i, l - s_hat)])
                g_temp[(i, l)][r - 1] = s_hat

            for l in range(2, min(k, (r - 1) * subtree_size(1) + 1) + 1):
                f[(i, l)] = f_temp[(i, l)]
                g[(i, l)] = g_temp[(i, l)]

    f[(0, 0)] = 0
    f[(0, 1)] = 0
    g[(0, 0)] = [0] * root_count
    g[(0, 1)] = [0] * root_count

    child_max_size = subtree_size(1) + 1

    for r in range(1, root_count + 1):
        current_max_l = min(k + 1, r * child_max_size + 1)
        for l in range(1, current_max_l + 1):
            capacity_before = 1 + (r - 1) * child_max_size
            s_minus = max(0, l - capacity_before)
            s_plus = min(l - 1, child_max_size)
            s_hat = max(
                range(s_minus, s_plus + 1),
                key=lambda s: f[(r, s)] + f[(0, l - s)],
            )

            f_temp[(0, l)] = f[(r, s_hat)] + f[0, l - s_hat]

            g_temp[(0, l)] = list(g[(0, l - s_hat)])
            g_temp[(0, l)][r - 1] = s_hat

        for l in range(1, current_max_l + 1):
            f[(0, l)] = f_temp[(0, l)]
            g[(0, l)] = g_temp[(0, l)]

    tau = MathArray([0] * n)
    gamma = MathArray([0] * n)

    virtual_budget = k + 1
    root_splits = g[(0, virtual_budget)]

    for r in range(1, root_count + 1):
        assigned_budget = root_splits[r - 1]

        if assigned_budget > 0:
            tau[r] = 1
            gamma[r] = assigned_budget
        else:
            tau[r] = 0
            gamma[r] = 0

    for j in range(max_level):
        if j == 0:
            start_node = 1
            end_node = root_count + 1
        else:
            start_node = root_count * (d ** (j - 1)) + 1
            end_node = root_count * (d ** j) + 1

        for i in range(start_node, end_node):
            if tau[i] == 1:
                current_budget = gamma[i]

                if (i, current_budget) not in g:
                    continue

                child_splits = g[(i, current_budget)]

                for r in range(1, d + 1):
                    budget_for_child = child_splits[r - 1]

                    if budget_for_child > 0:
                        if j == 0:
                            # Special mapping for Roots -> Level 1
                            if r == 2:
                                child_idx = i + root_count
                            else:
                                continue
                        else:
                            child_idx = d * (i - 1) + r

                        tau[child_idx] = 1
                        gamma[child_idx] = budget_for_child

    y_hat = MathArray([0] * n)
    for i in range(1, n + 1):
        y_hat[i] = y[i] * tau[i]

    return WtCoeffs.from_flat_coeffs(
        flat_coeffs=y_hat.data,
        root_count=root_count,
        max_level=max_level,
        wavelet=wt_coeffs.wavelet,
    )
