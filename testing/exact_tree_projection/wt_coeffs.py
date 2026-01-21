class WtCoeffs:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.max_level = len(coeffs) - 2
        self.root_count = len(coeffs[1])

    def get_coeffs(self):
        return self.coeffs

    def get_one_index_coeffs(self):
        raw_coeffs = []
        for level in self.coeffs:
            raw_coeffs.extend(level)
        return raw_coeffs

    def print_tree(self, print_approximation=False):
        if print_approximation:
            print(f"Approximation: {self.coeffs[0]}")
        for i in range(self.max_level):
            print(f"Level {i}:\t"+'\t'.join(map(str, self.coeffs[i+1])))
