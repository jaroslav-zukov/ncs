import numpy as np
import pywt


def main():
    print("Wavelet test")

    signal = np.arange(1024)

    wavelets = pywt.wavelist(kind='discrete')
    orthogonal_wavelets = [w for w in wavelets if pywt.Wavelet(w).orthogonal]

    print(f"Orthogonal wavelets ({len(orthogonal_wavelets)}): {orthogonal_wavelets}")

    wave = 'sym7'

    wt_signal = pywt.wavedec(signal, wavelet=wave, level=None, mode='periodization')
    print(f"Wavelet waves ({len(wt_signal)}): {[len(arr) for arr in wt_signal]}")


if __name__ == '__main__':
    main()