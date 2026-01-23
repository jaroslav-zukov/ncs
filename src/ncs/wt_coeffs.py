import numpy as np


class WtCoeffs:
    def __init__(self, coeff_groups, wavelet):
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

    @classmethod
    def from_flat_coeffs(cls, flat_coeffs, root_count, max_level, wavelet):
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
        return np.concatenate(self.coeff_groups)

    @property
    def support(self):
        return set(np.nonzero(self.flat_coeffs)[0])

    @property
    def n(self):
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
