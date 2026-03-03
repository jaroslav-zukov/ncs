"""
Prototype: Wavelet Packet best-basis reconstruction from time-domain subsampling.

This script validates Option C: using a Wavelet Packet (WP) best-basis as the
sparsifying transform instead of the standard DWT, in order to reduce the
coherence between the sparsity basis and plain time-domain subsampling.

Theory recap
------------
Standard DWT wavelets are time-localised → highly coherent with time-domain
subsampling (μ ≈ O(1)).  WP basis functions at decomposition depth j satisfy:

    max_t |ψ_{j,p}(t)| ≤ C · 2^{-j/2}  →  μ(S, Ψ_WP) ≤ C² · 2^{-j}

The Coifman-Wickerhauser best-basis algorithm selects the WP tree leaves that
minimise Shannon entropy of the signal's WP coefficients — the maximally
sparse WP representation.  This implicitly trades sparsity against coherence.

Experiment design
-----------------
Four reconstruction conditions compared across m ∈ [20, 500]:

  A. Subsampling + DWT (classical CoSaMP) — the coherent baseline
  B. Subsampling + WP best-basis (OMP) — Option C
  C. Gaussian + DWT (CoSaMP) — the incoherent gold standard
  D. Random modulation + DWT (CoSaMP) — Option B hardware baseline

Key insight for CoSaMP vs OMP
------------------------------
CoSaMP relies on a tree-structured sparse projection (DWT case).  For an
arbitrary WP best-basis, the coefficient index set has no canonical tree
structure.  We therefore use Orthogonal Matching Pursuit (OMP) for the WP
condition — flat k-sparse projection suffices since the basis already picks
the sparsest representation.

Coherence comparison
--------------------
The script also measures and reports empirical mutual coherence for:
  - Haar DWT basis
  - WP best-basis (signal-adaptive)
  - Fourier basis (theoretical reference)

Output
------
figures/wp_reconstruction_<timestamp>.png  — 4-condition comparison plot
figures/wp_coherence_<timestamp>.png       — coherence comparison bar chart
"""

import argparse
import os
import sys
from datetime import datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

# ── Measurement operators (inline, self-contained) ───────────────────────────

def make_subsampling_op(n, m, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    scale = np.sqrt(n / m)
    measure   = lambda x: x[idx] * scale
    adjoint   = lambda y: np.array([np.sum(y * (np.arange(m) == np.searchsorted(idx, i))) if i in idx else 0 for i in range(n)])
    # Faster adjoint:
    def _adj(y):
        out = np.zeros(n); out[idx] = y * scale; return out
    def _pinv(y):
        out = np.zeros(n); out[idx] = y / scale; return out
    return measure, _adj, _pinv, idx


def make_gaussian_op(n, m, seed=None):
    rng = np.random.default_rng(seed)
    Phi = rng.normal(0, 1 / np.sqrt(m), size=(m, n))
    Pinv = np.linalg.pinv(Phi)
    return (lambda x: Phi @ x,
            lambda y: Phi.T @ y,
            lambda y: Pinv @ y)


def make_modulation_op(n, m, seed=None):
    rng = np.random.default_rng(seed)
    chip = rng.choice([-1.0, 1.0], size=n)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    scale = np.sqrt(n / m)
    def measure(x): return (chip * x)[idx] * scale
    def adjoint(y): out = np.zeros(n); out[idx] = y * scale; return chip * out
    def pinv(y): out = np.zeros(n); out[idx] = y / scale; return chip * out
    return measure, adjoint, pinv


# ── DWT helpers ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _dwt_spec(n, wavelet):
    level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
    template = pywt.wavedec(np.zeros(n), wavelet=wavelet,
                             mode='periodization', level=level)
    arr, slices = pywt.coeffs_to_array(template)
    return arr.shape, slices, level


def dwt_forward(signal, wavelet='haar'):
    n = len(signal)
    shape, slices, level = _dwt_spec(n, wavelet)
    coeffs = pywt.wavedec(signal, wavelet=wavelet, mode='periodization', level=level)
    arr, _ = pywt.coeffs_to_array(coeffs)
    return arr.ravel()


def dwt_inverse(flat, wavelet='haar', n=None):
    shape, slices, level = _dwt_spec(n, wavelet)
    arr = flat.reshape(shape)
    coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec')
    return pywt.waverec(coeffs, wavelet=wavelet, mode='periodization')


# ── WP best-basis helpers (self-contained) ───────────────────────────────────

def shannon_entropy(c: np.ndarray) -> float:
    c2 = c ** 2
    s = c2.sum()
    if s < 1e-15:
        return 0.0
    p = c2 / s
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


def wp_best_basis(signal: np.ndarray, wavelet: str = 'haar', max_level: int = None):
    """
    Coifman-Wickerhauser best basis for a given signal.

    Returns list of (level, path) leaf nodes that minimise Shannon entropy.
    """
    n = len(signal)
    if max_level is None:
        max_level = int(np.log2(n))

    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet,
                             mode='periodization', maxlevel=max_level)

    # Collect all node data
    nodes = {}
    for lv in range(1, max_level + 1):
        for node in wp.get_level(lv, 'natural'):
            nodes[node.path] = node.data.copy()

    def recurse(path, lv):
        data = nodes.get(path)
        if data is None:
            return 0.0, []
        ent = shannon_entropy(data)
        if lv >= max_level:
            return ent, [(lv, path)]
        le, ll = recurse(path + 'a', lv + 1)
        re, rl = recurse(path + 'd', lv + 1)
        if ent <= le + re:
            return ent, [(lv, path)]
        return le + re, ll + rl

    _, al = recurse('a', 1)
    _, dl = recurse('d', 1)
    return al + dl


def wp_encode(signal: np.ndarray, leaf_nodes: list, wavelet: str, max_level: int) -> np.ndarray:
    n = len(signal)
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet,
                             mode='periodization', maxlevel=max_level)
    parts = []
    for (lv, path) in sorted(leaf_nodes, key=lambda x: x[1]):
        parts.append(wp[path].data.copy())
    return np.concatenate(parts)


def wp_decode(flat: np.ndarray, leaf_nodes: list, wavelet: str,
              max_level: int, n: int) -> np.ndarray:
    wp = pywt.WaveletPacket(data=np.zeros(n), wavelet=wavelet,
                             mode='periodization', maxlevel=max_level)
    sorted_leaves = sorted(leaf_nodes, key=lambda x: x[1])
    idx = 0
    for (lv, path) in sorted_leaves:
        size = n // (2 ** lv)
        wp[path] = flat[idx:idx + size]
        idx += size
    return wp.reconstruct(update=False)


# ── Sparse projection & support ───────────────────────────────────────────────

def hard_thresh(arr, k):
    out = np.zeros_like(arr)
    top = np.argsort(np.abs(arr))[-k:]
    out[top] = arr[top]
    return out


def support(arr):
    return set(np.nonzero(arr)[0])


# ── CoSaMP (flat sparsity, generic operators) ─────────────────────────────────

def cosamp(A, At, Apinv, y, k, n, max_iter=50):
    x = np.zeros(n)
    r = y.copy()
    for _ in range(max_iter):
        e = At(r)
        omega = np.nonzero(hard_thresh(e, 2 * k))[0]
        T = np.union1d(np.nonzero(x)[0], omega).astype(int)
        b = np.zeros(n)
        b[T] = Apinv(y)[T]
        x = hard_thresh(b, k)
        r = y - A(x)
        if np.linalg.norm(r) < 1e-10:
            break
    return x


# ── OMP ───────────────────────────────────────────────────────────────────────

def omp(A, At, y, k, n, max_iter=None):
    """
    Orthogonal Matching Pursuit via operator-based approach.

    Uses the adjoint (correlation) to greedily select atoms, then solves
    a least-squares system on the selected support.
    """
    if max_iter is None:
        max_iter = k
    x = np.zeros(n)
    r = y.copy()
    support_set = []

    # Precompute a matrix-free least-squares is hard without the explicit matrix.
    # We use a small-scale explicit Gram matrix on the growing support instead.
    # Build columns of A on demand by probing with unit vectors.

    # For efficiency we probe A with unit vectors only for the support indices.
    def col(i):
        e = np.zeros(n); e[i] = 1.0; return A(e)

    for _ in range(max_iter):
        corr = At(r)
        # Pick atom with largest correlation, not already selected
        corr[support_set] = 0
        new_atom = int(np.argmax(np.abs(corr)))
        support_set.append(new_atom)

        # Least-squares on support: build Phi_T (m x |T|) column by column
        T = support_set
        Phi_T = np.column_stack([col(i) for i in T])
        # Solve min ||Phi_T @ c - y||
        c, _, _, _ = np.linalg.lstsq(Phi_T, y, rcond=None)
        x = np.zeros(n)
        x[T] = c
        r = y - A(x)
        if np.linalg.norm(r) < 1e-8:
            break

    return x


# ── Coherence measurement ─────────────────────────────────────────────────────

def empirical_coherence_dwt(n: int, wavelet: str = 'haar', n_basis: int = 30) -> float:
    """μ(time-subsampling, DWT) = max_j max_t |ψ_j(t)|²."""
    rng = np.random.default_rng(0)
    max_coh = 0.0
    for i in rng.choice(n, size=n_basis, replace=False):
        e = np.zeros(n); e[i] = 1.0
        c = dwt_forward(e, wavelet)
        max_coh = max(max_coh, np.max(c ** 2))
    return max_coh


def empirical_coherence_wp(leaf_nodes: list, wavelet: str, max_level: int,
                            n: int, n_basis: int = 30) -> float:
    """μ(time-subsampling, WP best-basis) via random basis function probing."""
    rng = np.random.default_rng(0)
    max_coh = 0.0
    for (lv, path) in leaf_nodes:
        size = n // (2 ** lv)
        n_probe = min(size, max(1, n_basis // len(leaf_nodes)))
        for k in rng.choice(size, size=n_probe, replace=False):
            flat = np.zeros(n)
            idx = 0
            for (l2, p2) in sorted(leaf_nodes, key=lambda x: x[1]):
                sz2 = n // (2 ** l2)
                if (l2, p2) == (lv, path):
                    flat[idx + (k % sz2)] = 1.0
                idx += sz2
            basis_fn = wp_decode(flat, leaf_nodes, wavelet, max_level, n)
            max_coh = max(max_coh, float(np.max(basis_fn ** 2)))
    return max_coh


def empirical_coherence_fourier(n: int, n_basis: int = 30) -> float:
    """μ(time-subsampling, DFT) ≈ 1/n (theoretical)."""
    # Fourier basis vector: |F_{k,t}|² = 1/n for all k,t
    return 1.0 / n


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_dwt_sparse_signal(n: int, s: int, wavelet: str = 'haar') -> np.ndarray:
    """
    Generate a signal that is s-sparse in the DWT domain.

    Produces a flat wavelet coefficient vector with s random non-zero entries,
    then synthesises the time-domain signal.
    """
    coeffs = np.zeros(n)
    idx = np.random.choice(n, size=s, replace=False)
    coeffs[idx] = np.random.randint(-300, 300, size=s).astype(float)
    return dwt_inverse(coeffs, wavelet=wavelet, n=n)


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(n=512, s=8, n_signals=8, m_values=None, wavelet='haar',
                   show_plot=True):
    if m_values is None:
        m_values = np.linspace(20, 400, 16).astype(int)

    max_level = int(np.log2(n))
    results = []

    # ── Pre-compute WP best basis on a representative signal ──────────────────
    print("Computing WP best basis on a representative signal...")
    rep_signal = generate_dwt_sparse_signal(n, s, wavelet)
    leaf_nodes = wp_best_basis(rep_signal, wavelet=wavelet, max_level=max_level)
    n_leaves = len(leaf_nodes)
    avg_level = np.mean([lv for lv, _ in leaf_nodes])
    print(f"  Best basis: {n_leaves} leaves, mean level = {avg_level:.2f}")

    # ── Coherence comparison ──────────────────────────────────────────────────
    print("\nMeasuring coherence...")
    mu_dwt = empirical_coherence_dwt(n, wavelet)
    mu_wp = empirical_coherence_wp(leaf_nodes, wavelet, max_level, n)
    mu_fourier = empirical_coherence_fourier(n)
    print(f"  μ(DWT)     = {mu_dwt:.6f}")
    print(f"  μ(WP best) = {mu_wp:.6f}")
    print(f"  μ(Fourier) = {mu_fourier:.6f}")
    print(f"  WP coherence reduction vs DWT: {mu_dwt / mu_wp:.2f}x")

    # ── Reconstruction experiments ────────────────────────────────────────────
    print(f"\nRunning reconstruction for {n_signals} signals × {len(m_values)} m values...")

    for m in tqdm(m_values, desc="m sweep"):
        for trial in range(n_signals):
            signal = generate_dwt_sparse_signal(n, s, wavelet)
            dwt_coeffs_true = dwt_forward(signal, wavelet)
            wp_coeffs_true = wp_encode(signal, leaf_nodes, wavelet, max_level)

            # ── Condition A: Subsampling + DWT (CoSaMP) ───────────────────────
            A_meas, A_adj, A_pinv, _ = make_subsampling_op(n, m, seed=trial)

            def phi_A(x_flat):
                sig = dwt_inverse(x_flat, wavelet=wavelet, n=n)
                return A_meas(sig)
            def phiT_A(y):
                upsampled = A_adj(y)
                return dwt_forward(upsampled, wavelet)
            def phiPinv_A(y):
                upsampled = A_pinv(y)
                return dwt_forward(upsampled, wavelet)

            y_A = phi_A(dwt_coeffs_true)
            x_hat_A = cosamp(phi_A, phiT_A, phiPinv_A, y_A, s, n)
            err_A = np.linalg.norm(dwt_coeffs_true - x_hat_A) / (np.linalg.norm(dwt_coeffs_true) + 1e-15)
            results.append({'condition': 'A: Subsample+DWT', 'm': m, 'rel_error': err_A,
                            'missed': len(support(dwt_coeffs_true) - support(x_hat_A))})

            # ── Condition B: Subsampling + WP best-basis (OMP) ───────────────
            B_meas, B_adj, B_pinv, _ = make_subsampling_op(n, m, seed=trial)

            def phi_B(x_flat_wp):
                sig = wp_decode(x_flat_wp, leaf_nodes, wavelet, max_level, n)
                return B_meas(sig)
            def phiT_B(y):
                upsampled = B_adj(y)
                return wp_encode(upsampled, leaf_nodes, wavelet, max_level)

            y_B = phi_B(wp_coeffs_true)
            x_hat_B = omp(phi_B, phiT_B, y_B, s, n, max_iter=s)
            err_B = np.linalg.norm(wp_coeffs_true - x_hat_B) / (np.linalg.norm(wp_coeffs_true) + 1e-15)
            results.append({'condition': 'B: Subsample+WP', 'm': m, 'rel_error': err_B,
                            'missed': len(support(wp_coeffs_true) - support(x_hat_B))})

            # ── Condition C: Gaussian + DWT (CoSaMP) ─────────────────────────
            C_meas, C_adj, C_pinv = make_gaussian_op(n, m, seed=trial)

            def phi_C(x_flat):
                sig = dwt_inverse(x_flat, wavelet=wavelet, n=n)
                return C_meas(sig)
            def phiT_C(y):
                return dwt_forward(C_adj(y), wavelet)
            def phiPinv_C(y):
                return dwt_forward(C_pinv(y), wavelet)

            y_C = phi_C(dwt_coeffs_true)
            x_hat_C = cosamp(phi_C, phiT_C, phiPinv_C, y_C, s, n)
            err_C = np.linalg.norm(dwt_coeffs_true - x_hat_C) / (np.linalg.norm(dwt_coeffs_true) + 1e-15)
            results.append({'condition': 'C: Gaussian+DWT', 'm': m, 'rel_error': err_C,
                            'missed': len(support(dwt_coeffs_true) - support(x_hat_C))})

            # ── Condition D: Random Modulation + DWT (CoSaMP) ─────────────────
            D_meas, D_adj, D_pinv = make_modulation_op(n, m, seed=trial)

            def phi_D(x_flat):
                sig = dwt_inverse(x_flat, wavelet=wavelet, n=n)
                return D_meas(sig)
            def phiT_D(y):
                return dwt_forward(D_adj(y), wavelet)
            def phiPinv_D(y):
                return dwt_forward(D_pinv(y), wavelet)

            y_D = phi_D(dwt_coeffs_true)
            x_hat_D = cosamp(phi_D, phiT_D, phiPinv_D, y_D, s, n)
            err_D = np.linalg.norm(dwt_coeffs_true - x_hat_D) / (np.linalg.norm(dwt_coeffs_true) + 1e-15)
            results.append({'condition': 'D: RandMod+DWT', 'm': m, 'rel_error': err_D,
                            'missed': len(support(dwt_coeffs_true) - support(x_hat_D))})

    df = pd.DataFrame(results)

    # ── Plot 1: Reconstruction comparison ────────────────────────────────────
    os.makedirs('figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    fig, ax = plt.subplots(figsize=(13, 7))
    palette = {
        'A: Subsample+DWT': '#e74c3c',
        'B: Subsample+WP':  '#2ecc71',
        'C: Gaussian+DWT':  '#3498db',
        'D: RandMod+DWT':   '#f39c12',
    }
    sns.lineplot(data=df, x='m', y='rel_error', hue='condition',
                 estimator='mean', errorbar='sd', palette=palette,
                 linewidth=2.2, ax=ax)
    ax.set_xlabel('Number of measurements m', fontsize=12)
    ax.set_ylabel('Relative ℓ₂ error', fontsize=12)
    ax.set_title(
        f'Wavelet Packet Option C — 4-way reconstruction comparison\n'
        f'n={n}, sparsity s={s}, wavelet={wavelet}', fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    info = (f'n={n}, s={s}, {n_signals} trials\n'
            f'μ(DWT)={mu_dwt:.4f}\n'
            f'μ(WP) ={mu_wp:.4f}  ({mu_dwt/mu_wp:.1f}x lower)\n'
            f'μ(FFT)={mu_fourier:.4f}')
    at = AnchoredText(info, loc='upper right', frameon=True,
                      bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax.add_artist(at)

    plt.tight_layout()
    fname1 = f'figures/wp_reconstruction_{timestamp}.png'
    plt.savefig(fname1, dpi=150)
    print(f'\nSaved: {fname1}')
    if show_plot:
        plt.show()

    # ── Plot 2: Coherence bar chart ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    labels = ['DWT (Haar)', f'WP Best Basis\n(depth≈{avg_level:.1f})', 'Fourier (DFT)']
    values = [mu_dwt, mu_wp, mu_fourier]
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax2.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                 f'{v:.5f}', ha='center', fontsize=11)
    ax2.set_ylabel('Empirical mutual coherence μ(S, Ψ)', fontsize=12)
    ax2.set_title(
        f'Coherence comparison: DWT vs WP vs Fourier\nn={n}, {wavelet} wavelet', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fname2 = f'figures/wp_coherence_{timestamp}.png'
    plt.savefig(fname2, dpi=150)
    print(f'Saved: {fname2}')
    if show_plot:
        plt.show()

    # ── Summary statistics ────────────────────────────────────────────────────
    print('\n── Summary (mean relative error at highest m) ──')
    high_m = df['m'].max()
    for cond in df['condition'].unique():
        subset = df[(df['condition'] == cond) & (df['m'] == high_m)]
        print(f"  {cond:<25}  mean err = {subset['rel_error'].mean():.4f}")

    return {
        'mu_dwt': mu_dwt,
        'mu_wp': mu_wp,
        'mu_fourier': mu_fourier,
        'coherence_reduction': mu_dwt / mu_wp,
        'df': df,
        'leaf_nodes': leaf_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description='WP Reconstruction (Option C)')
    parser.add_argument('--no-show', action='store_true',
                        help='Disable interactive plot display.')
    parser.add_argument('--n', type=int, default=512, help='Signal length (power of 2).')
    parser.add_argument('--s', type=int, default=8, help='Sparsity level.')
    parser.add_argument('--signals', type=int, default=8,
                        help='Number of random signals per m value.')
    parser.add_argument('--wavelet', type=str, default='haar',
                        help='Orthogonal wavelet (haar, db4, db8, ...).')
    args = parser.parse_args()

    print('=' * 60)
    print('Wavelet Packet Reconstruction — Option C Validation')
    print('=' * 60)
    m_values = np.linspace(20, min(args.n - 1, 400), 16).astype(int)

    run_experiment(
        n=args.n,
        s=args.s,
        n_signals=args.signals,
        m_values=m_values,
        wavelet=args.wavelet,
        show_plot=not args.no_show,
    )


if __name__ == '__main__':
    main()
