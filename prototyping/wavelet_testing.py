import numpy as np
import pywt

from ncs.wt_coeffs import WtCoeffs


def main():
    print("Wavelet test")

    signal = np.arange(8)

    wavelets = pywt.wavelist(kind="discrete")
    orthogonal_wavelets = [w for w in wavelets if pywt.Wavelet(w).orthogonal]

    print(f"Orthogonal wavelets ({len(orthogonal_wavelets)}): {orthogonal_wavelets}")

    wave = "db2"

    wt_coeff_groups = pywt.wavedec(
        signal, wavelet=wave, level=None, mode="periodization"
    )
    print(
        f"Wavelet waves ({len(wt_coeff_groups)}): {[len(arr) for arr in wt_coeff_groups]}"
    )

    wt_coeffs = WtCoeffs(wt_coeff_groups, wave)
    print(f"Max level {wt_coeffs.max_level}")
    print(f"Root count {wt_coeffs.root_count}")
    all_coeffs = wt_coeffs.flat_coeffs
    print(f"All coeffs length: {len(all_coeffs)}")

    flat_coeffs = np.zeros_like(all_coeffs)
    flat_coeffs[0] = all_coeffs[0]

    for group in wt_coeff_groups:
        for coeff in group:
            print(f"{coeff:.1f},")

    # second_constructor_coeffs = WtCoeffs.from_flat_coeffs(
    #     flat_coeffs, wt_coeffs.root_count, wt_coeffs.max_level, wave
    # )

    # for level in range(wt_coeffs.max_level+1):
    #     print(f"Level {level}: len original coeffs: {len(wt_coeffs.coeff_groups[level])}, new coeffs: {len(second_constructor_coeffs.coeff_groups[level])}")
    #     print(f"Elemenwise equal: {np.all(wt_coeffs.coeff_groups[level] == second_constructor_coeffs.coeff_groups[level])}")

    # projected_wt = tree_projection(wt_coeffs, 1)
    # print(f"Project onto itself: {second_constructor_coeffs == projected_wt}")
    # TODO: make a proper test like this for each component

    # print(f"N: {projected_wt.n}")


if __name__ == "__main__":
    main()
