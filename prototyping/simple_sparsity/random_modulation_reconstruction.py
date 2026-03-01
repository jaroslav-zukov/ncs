"""
Random Modulation + Time Subsampling (Option B) — Validation Script

Theoretical prediction (Tropp et al., 2010):
  - Signal sparse in WAVELET domain → time-domain signal is NOT time-localized
  - Plain subsampling S·x is COHERENT with wavelet basis → RIP fails → bad recovery
  - Random modulation S·(r⊙x), r∈{±1}^n, is INCOHERENT with wavelets → RIP holds → good recovery

This script:
1. Generates signals that are SPARSE IN WAVELET DOMAIN
   (i.e., only s wavelet coefficients are non-zero; time-domain signal is spread out)
2. Applies three measurement schemes to the TIME-DOMAIN signal
3. Runs CoSaMP in wavelet-coefficient domain
4. Compares Gaussian vs Plain Subsampling vs Random Modulation

Expected: Gaussian ≈ Random Modulation >> Plain Subsampling (in terms of recovery quality)
"""
import argparse
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns

from ncs.measurement_module import (
    create_gaussian_operator,
    create_random_modulation_operator,
    create_subsampling_operator,
)


# ── Wavelet helpers ─────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _wt_spec(n: int, wavelet: str):
    level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    template = pywt.wavedec(np.zeros(n), wavelet=wavelet, mode="periodization", level=level)
    _, slices = pywt.coeffs_to_array(template)
    shape = pywt.coeffs_to_array(template)[0].shape
    return shape, slices, level


def flat_to_time(flat_coeffs: np.ndarray, n: int, wavelet: str = "db4") -> np.ndarray:
    """Flat wavelet coefficients → time-domain signal."""
    shape, slices, level = _wt_spec(n, wavelet)
    arr = flat_coeffs.reshape(shape)
    coeffs_list = pywt.array_to_coeffs(arr, slices, output_format="wavedec")
    return pywt.waverec(coeffs_list, wavelet=wavelet, mode="periodization")


def time_to_flat(x: np.ndarray, n: int, wavelet: str = "db4") -> np.ndarray:
    """Time-domain signal → flat wavelet coefficients."""
    _, _, level = _wt_spec(n, wavelet)
    coeffs_list = pywt.wavedec(x, wavelet=wavelet, mode="periodization", level=level)
    arr, _ = pywt.coeffs_to_array(coeffs_list)
    return arr.ravel()


# ── CoSaMP helpers ───────────────────────────────────────────────────────────

def sparse_projection(v: np.ndarray, s: int) -> np.ndarray:
    out = np.zeros_like(v)
    idx = np.argsort(np.abs(v))[-s:]
    out[idx] = v[idx]
    return out


def cosamp(phi, phi_T, phi_pinv, y, s, n, max_iter=50):
    x = np.zeros(n)
    r = y.copy()
    for _ in range(max_iter):
        e = phi_T(r)
        omega = np.nonzero(sparse_projection(e, 2 * s))[0]
        sup = np.union1d(np.nonzero(x)[0], omega).astype(int)
        b = phi_pinv(y)
        b_proj = np.zeros(n)
        b_proj[sup] = b[sup]
        x = sparse_projection(b_proj, s)
        r = y - phi(x)
        if np.linalg.norm(r) < 1e-10:
            break
    return x


# ── Experiment ───────────────────────────────────────────────────────────────

def run_experiment(n: int, s: int, m_values, n_trials: int, wavelet: str = "db4"):
    shape, slices, level = _wt_spec(n, wavelet)
    n_coeffs = int(np.prod(shape))
    rng = np.random.default_rng(2024)

    records = []

    for m in m_values:
        print(f"  m={m}", end="", flush=True)
        for trial in range(n_trials):
            # ── Generate wavelet-sparse signal ──────────────────────────────
            # Sparse in WAVELET domain → non-localized in TIME domain
            sparse_wt = np.zeros(n_coeffs)
            idx = rng.choice(n_coeffs, size=s, replace=False)
            sparse_wt[idx] = rng.integers(-100, 100, size=s).astype(float)
            x_time = flat_to_time(sparse_wt, n, wavelet)  # time-domain signal

            for mode in ("gaussian", "subsampling", "random_modulation"):
                seed = int(rng.integers(0, 2**31))

                if mode == "gaussian":
                    phi_mat, phi_T_mat, phi_pinv_mat = create_gaussian_operator(n_coeffs, m, seed)
                    # Gaussian works in wavelet-coeff domain directly
                    y = phi_mat(sparse_wt)

                    def phi(w):      return phi_mat(w)
                    def phi_T(y_):   return phi_T_mat(y_)
                    def phi_pinv(y_): return phi_pinv_mat(y_)

                elif mode == "subsampling":
                    meas_op, adj_op, pinv_op = create_subsampling_operator(n, m, seed)
                    y = meas_op(x_time)

                    def phi(w, _m=meas_op):     return _m(flat_to_time(w, n, wavelet))
                    def phi_T(y_, _a=adj_op):   return time_to_flat(_a(y_), n, wavelet)
                    def phi_pinv(y_, _p=pinv_op): return time_to_flat(_p(y_), n, wavelet)

                else:  # random_modulation
                    meas_op, adj_op, pinv_op = create_random_modulation_operator(n, m, seed)
                    y = meas_op(x_time)

                    def phi(w, _m=meas_op):     return _m(flat_to_time(w, n, wavelet))
                    def phi_T(y_, _a=adj_op):   return time_to_flat(_a(y_), n, wavelet)
                    def phi_pinv(y_, _p=pinv_op): return time_to_flat(_p(y_), n, wavelet)

                wt_hat = cosamp(phi, phi_T, phi_pinv, y, s, n_coeffs)
                err = np.linalg.norm(sparse_wt - wt_hat) / (np.linalg.norm(sparse_wt) + 1e-12)
                records.append({"m": m, "mode": mode, "relative_error": float(err)})

        print(" done")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    n = 1024
    s = 10
    n_trials = 5
    m_values = np.linspace(50, 600, 12).astype(int)

    print(f"=== Random Modulation Validation  n={n}, s={s}, trials={n_trials} ===")
    df = run_experiment(n, s, m_values, n_trials)

    label_map = {
        "gaussian": "Gaussian",
        "subsampling": "Plain Subsampling",
        "random_modulation": "Random Modulation (Option B)",
    }
    df["Mode"] = df["mode"].map(label_map)

    print("\n=== Mean Relative Error ===")
    summary = df.groupby("Mode")["relative_error"].mean()
    for mode, err in summary.items():
        print(f"  {mode}: {err:.4f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    palette = {
        "Gaussian": "#2196F3",
        "Plain Subsampling": "#F44336",
        "Random Modulation (Option B)": "#4CAF50",
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    for mode_label, color in palette.items():
        sub = df[df["Mode"] == mode_label]
        sns.lineplot(
            data=sub, x="m", y="relative_error",
            estimator="mean", color=color, linewidth=2.5,
            errorbar="sd", label=mode_label, ax=ax,
        )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Error = 1 (no recovery)")
    ax.set_xlabel("Number of Measurements (m)", fontsize=13)
    ax.set_ylabel("Relative Error  ‖ŵ − w‖ / ‖w‖", fontsize=13)
    ax.set_title(
        "CS Recovery of Wavelet-Sparse Signals\n"
        "Gaussian vs Plain Subsampling vs Random Modulation (Option B)\n"
        f"n={n}, s={s}, {n_trials} trials — wavelet domain recovery",
        fontsize=13,
    )
    ax.set_ylim(bottom=0, top=1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    outpath = "figures/random_modulation_vs_subsampling_comparison.png"
    plt.savefig(outpath, dpi=300)
    print(f"\nSaved plot to {outpath}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
