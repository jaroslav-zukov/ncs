"""
Wavelet coefficient container with dyadic tree structure.

This module provides WtCoeffs, the central data structure for wavelet
coefficient vectors throughout the NCS pipeline.

Coefficient layouts
-------------------
grouped layout (coeff_groups)
    A Python list of numpy arrays, one per decomposition level, ordered
    from coarsest (level 0, approximation / scaling coefficients) to
    finest (level max_level, detail coefficients at the smallest scale):

        coeff_groups[0]          — root_count approximation coefficients
        coeff_groups[1]          — root_count * 1  detail coefficients
        coeff_groups[2]          — root_count * 2  detail coefficients
        …
        coeff_groups[max_level]  — root_count * 2^(max_level-1) detail coeffs

    This matches pywt.wavedec output format exactly and is used for
    pywt.waverec inverse reconstruction.

flat layout (flat_coeffs)
    The concatenation of all coeff_groups into a single numpy array of
    length n = root_count * 2^max_level.  This flat vector is the input
    to tree_projection and to the measurement operators (Φ · flat_coeffs).

Dyadic tree structure
---------------------
The wavelet coefficient tree is a d=2-ary rooted forest.  Roots are the
root_count approximation coefficients (indices 1 … root_count in 1-based
MathArray notation).  Each node i at level j (0-based, root = level 0)
has exactly two children at level j+1:
    child 1: 2*(i-1)+1  (in 1-based indexing within the level)
    child 2: 2*(i-1)+2

The special level-0 → level-1 connection in the NCS implementation treats
each root i as having a single level-1 child at index i + root_count
(because level 1 has the same count as level 0).  Subsequent levels follow
the standard 2*(i-1)+r indexing.

A connected subtree of size k rooted at a subset of roots contains nodes
that form a connected path from a root down to leaves, used by the exact
tree projection algorithm.
"""

import numpy as np


class WtCoeffs:
    """
    Container for discrete wavelet transform coefficients with tree structure.

    Stores both the grouped (level-by-level) and flat representations of a
    wavelet coefficient vector and exposes the tree parameters (root_count,
    max_level) needed by the exact tree projection algorithm and by the
    measurement operators.

    Attributes:
        coeff_groups (list[np.ndarray]): Grouped coefficients ordered from
            coarsest (index 0) to finest (index max_level) decomposition
            level.  Compatible with pywt.waverec.
        wavelet (str): Name of the orthogonal wavelet used to produce these
            coefficients (e.g. 'haar', 'db4').
        max_level (int): Number of decomposition levels = len(coeff_groups)-1.
        root_count (int): Number of coarsest-scale approximation coefficients
            = len(coeff_groups[0]).  Determines the branching of the tree.
    """
    def __init__(self, coeff_groups, wavelet):
        """
        Construct a WtCoeffs object from grouped wavelet coefficients.

        Validates that the coefficient groups form a valid dyadic tree: the
        leaf level must have root_count * 2^(max_level-1) coefficients, as
        required by periodization-mode pywt.wavedec.

        Args:
            coeff_groups: List of numpy arrays [cA_n, cD_n, cD_{n-1}, …,
                cD_1] as returned by pywt.wavedec with mode="periodization".
                cA_n (index 0) is the coarsest approximation; cD_1 (last) is
                the finest detail level.
            wavelet: Orthogonal wavelet name string (e.g. 'haar').

        Raises:
            ValueError: If coeff_groups is empty or if the leaf-level count
                does not match the expected dyadic tree size.
            TypeError: If coeff_groups is not a list.
        """
        if not coeff_groups:
            raise ValueError("coeff_groups cannot be empty")

        if not isinstance(coeff_groups, list):
            raise TypeError("coeff_groups must be a list")

        self.coeff_groups = coeff_groups
        self.wavelet = wavelet
        self.max_level = len(coeff_groups) - 1
        self.root_count = len(coeff_groups[0])

        expected_leaf_count = self.root_count * 2 ** (self.max_level - 1)
        actual_leaf_count = len(self.coeff_groups[self.max_level])

        if actual_leaf_count != expected_leaf_count:
            raise ValueError(
                f"Invalid leaf count at level {self.max_level}: "
                f"expected {expected_leaf_count}, got {actual_leaf_count}. "
                "Try setting pywt.wavedec mode='periodization'"
            )

        # TODO: Hide the validation into a separate function, add the root size validation based on the wavelet filter length

    @classmethod
    def from_flat_coeffs(cls, flat_coeffs, root_count, max_level, wavelet):
        """
        Construct a WtCoeffs object from a flat coefficient vector.

        Splits the flat vector into per-level arrays according to the dyadic
        tree structure (root_count * 2^(level-1) coefficients per level ≥ 1,
        and root_count at level 0) and delegates to __init__.

        This is the primary factory used after tree projection and pseudo-
        inverse application, which work with flat arrays.

        Args:
            flat_coeffs: 1-D array-like of length root_count * 2^max_level.
            root_count: Number of coarsest-scale approximation coefficients.
            max_level: Number of decomposition levels.
            wavelet: Orthogonal wavelet name string.

        Returns:
            WtCoeffs: New instance with the reconstructed coeff_groups.

        Raises:
            ValueError: If len(flat_coeffs) != root_count * 2^max_level.
        """
        if len(flat_coeffs) != root_count * (2**max_level):
            raise ValueError(
                f"Invalid flat_coeffs length: expected {root_count * (2**max_level)}, got {len(flat_coeffs)}"
            )

        coeff_groups = []
        idx = 0

        for level in range(max_level + 1):
            level_size = root_count * (2 ** (level - 1)) if level > 0 else root_count
            coeff_groups.append(np.array(flat_coeffs[idx : idx + level_size]))
            idx += level_size

        return cls(coeff_groups, wavelet)

    def __eq__(self, other):
        """
        Test element-wise equality of two WtCoeffs objects.

        Returns True iff other is a WtCoeffs with the same max_level,
        root_count, and identical coefficient arrays at every level.
        Wavelet name is not compared (only structure and values).

        Args:
            other: Object to compare with.

        Returns:
            bool: True if all coefficient arrays are equal.
        """
        if not isinstance(other, WtCoeffs):
            return False
        if self.max_level != other.max_level or self.root_count != other.root_count:
            return False

        for level in range(self.max_level + 1):
            if not np.array_equal(self.coeff_groups[level], other.coeff_groups[level]):
                return False

        return True

    @property
    def flat_coeffs(self):
        """
        Concatenated coefficient vector of length n = root_count * 2^max_level.

        Obtained by np.concatenate(coeff_groups), ordering coefficients from
        coarsest scale (approximation) to finest scale (detail).  This flat
        vector is the representation used by measurement operators and the
        tree projection DP algorithm.

        Returns:
            np.ndarray: 1-D array of length n.
        """
        return np.concatenate(self.coeff_groups)

    @property
    def support(self):
        """
        Set of flat indices where the coefficient value is non-zero.

        Used in CoSaMP to track and merge the supports of successive
        estimates.  Indices correspond to positions in flat_coeffs.

        Returns:
            set[int]: 0-based flat indices with non-zero coefficients.
        """
        return set(np.nonzero(self.flat_coeffs)[0])

    @property
    def n(self):
        """
        Total number of wavelet coefficients: root_count * 2^max_level.

        Equals the length of the original signal (for periodization-mode DWT
        with power-of-2 signal lengths).

        Returns:
            int: Length of the flat coefficient vector.
        """
        return len(self.flat_coeffs)

    def on_support(self, support):
        """
        Create a copy with values only at specified support indices.

        Args:
            support: Set or array of indices where values should be non-zero

        Returns:
            New WtCoeffs with values only at support indices
        """
        flat_coeffs = np.zeros(self.n)
        support_indices = list(support)
        flat_coeffs[support_indices] = self.flat_coeffs[support_indices]

        return WtCoeffs.from_flat_coeffs(
            flat_coeffs=flat_coeffs,
            root_count=self.root_count,
            max_level=self.max_level,
            wavelet=self.wavelet,
        )
