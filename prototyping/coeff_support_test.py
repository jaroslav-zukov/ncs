from src.ncs.wt_coeffs import WtCoeffs


def main():
    coeffs = WtCoeffs([[4], [0], [1, 2]], "haar")
    print(f"Flat coefficients: {coeffs.flat_coeffs}")
    print(f"Coeff support: {coeffs.support}")
    # print(coeffs.support | {2})


if __name__ == "__main__":
    main()
