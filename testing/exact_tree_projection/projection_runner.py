from testing.exact_tree_projection.projection_algorithm import project
from testing.exact_tree_projection.wt_coeffs import WtCoeffs


def main():
    # coeffs = WtCoeffs([[1], [2], [3, 4]])
    coeffs = WtCoeffs([[0],[4], [1], [2, 3], [4, 5, 6, 7]]) # experiment 2
    # experiment 3 - two roots, simple trees
    print(coeffs.get_one_index_coeffs())
    coeffs.print_tree()

    project(coeffs, 2)

if __name__ == '__main__':
    main()