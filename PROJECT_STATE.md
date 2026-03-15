# Nanopore Compressive Sensing (NCS) — Project State

**Date:** 2026-03-15
**Author role:** PhD researcher building a model-based compressive sensing pipeline for Oxford Nanopore Technologies ionic current signals.

---

## 1. Project Goal

Apply compressive sensing to nanopore DNA sequencing ("squiggle") signals to reduce the data acquisition rate below 50% of Nyquist, ultimately targeting sub-30%. The pipeline uses **tree-sparse wavelet representations** reconstructed via **model-based CoSaMP** with **exact tree projection**, paired with hardware-feasible measurement operators that are incoherent with the wavelet basis.

The preprint is `Nanopore_Compressive_Sensing.pdf` in the project files.

---

## 2. Theoretical Framework

### 2.1 Core Pipeline

```
Signal x(t)  →  DWT (coif9)  →  wavelet coeffs α  →  tree-k-sparse model
                                                          ↓
Measurements y = Φ·x(t)     ←  measurement operator Φ    ↓
                                                          ↓
Reconstruction:  y  →  model-based CoSaMP  →  α̂  →  IDWT  →  x̂(t)
```

The pipeline operates in two domains:
- **Wavelet domain:** where sparsity is enforced (tree projection, CoSaMP iteration)
- **Time/frequency/sequency domain:** where physical measurements happen

The composition is: `Φ_effective = Φ_physical ∘ Ψ⁻¹` where `Ψ⁻¹` is the inverse DWT (synthesis). CoSaMP needs `phi(wt_coeffs)`, `phi_transpose(y)`, and `phi_pseudoinverse(y)` that compose the measurement operator with the DWT/IDWT.

### 2.2 Model-Based CS (Baraniuk et al. 2010)

Standard CS requires `m = O(k·log(N/k))` measurements for k-sparse signals of length N. Model-based CS with tree sparsity reduces this to `m = O(k)` by exploiting the fact that the number of valid tree-k-sparse supports is `O(C^k)` instead of `(N choose k)`. This eliminates the `log(N/k)` penalty entirely.

The **Model-RIP** guarantees isometry only over the union of tree-structured subspaces `M_k`, not all k-sparse subspaces. The **Restricted Amplification Property (RAmP)** handles compressible (not exactly sparse) signals by bounding how much the measurement matrix amplifies the non-modeled tail.

### 2.3 Asymptotic Incoherence (Adcock, Hansen & Poon)

**The coherence barrier:** Fourier-wavelet and Walsh-wavelet pairs have global coherence `μ(U) = O(1)`, which classically requires `m = O(N)` measurements — no compression. But this coherence is concentrated in a leading submatrix (low frequencies × coarse wavelet scales).

**Theorem 6.1:** The coherence of the matrix tails decays to zero:
`μ(P_K^⊥ U), μ(U P_K^⊥) = O(N⁻¹) as N → ∞`
This is **asymptotic incoherence** — the system is incoherent outside the low-frequency/coarse-scale block.

**Local coherence** `μ_{N,M}(k,l)` quantifies the interaction between frequency band k and wavelet scale l. Its decay is governed by:
- **Fine-to-coarse (j ≥ k+1):** decays as `2^{-ν·ΔR}` where ν = vanishing moments
- **Coarse-to-fine (j ≤ k-1):** decays as `2^{-(α-0.5)·ΔR}` where α = smoothness
- **Diagonal (j = k):** bounded by `1/N_{k-1}`

### 2.4 Multilevel Sampling (Theorem 6.2)

The optimal per-band measurement allocation:

```
m_k ≳ C · (N_k - N_{k-1})/N_{k-1} · (ŝ_k + Σ_{l<k} s_l·2^{-(α-0.5)·A_{k,l}} + Σ_{l>k} s_l·2^{-ν·B_{k,l}}) · log(Ñ)
```

Where:
- `s_l` = local sparsity at wavelet scale l
- `A_{k,l} = R_{k-1} - R_l`, `B_{k,l} = R_{l-1} - R_k` (dyadic scale differences)
- `ŝ_k = max{s_{k-1}, s_k, s_{k+1}}` (local sparsity including neighbors)

### 2.5 Walsh-Hadamard as Binary Fourier

The Walsh-Hadamard transform (WHT) is the binary analogue of the Fourier transform. Walsh functions are ±1 valued, ordered by **sequency** (number of zero-crossings = analogue of frequency). The Walsh-wavelet pair exhibits the **same asymptotic incoherence structure** as Fourier-wavelet, but in **dyadic blocks** rather than continuous bands.

Key difference from Fourier: coherence decay in Walsh-wavelet is governed by **piecewise-constant alignment** rather than smooth oscillatory cancellation. Less sensitive to wavelet smoothness α beyond a threshold, but strongly benefits from the block-diagonal structure when using sequency ordering.

Computational advantage: O(N log N) via fast WHT, storage O(2^q) instead of O(NM).

### 2.6 Tree Sparsity + Multilevel Synergy (Unpublished Gap)

No published paper formally combines model-RIP with the Adcock multilevel framework. The closest bridges are:
- **RIP in levels** (Bastounis & Hansen): generalizes RIP to accommodate asymptotic incoherence + structured sparsity
- **"One RIP to rule them all"** (Traonmilin & Gribonval): uniform recovery with structured models in continuous sensing

The key insight: tree-k-sparse signals have local sparsities `s_l` that decay rapidly at fine scales (energy concentrated at coarse scales). When fed into the multilevel formula, this collapses the high-frequency sampling budget. **This synergy is a core contribution of our paper.**

---

## 3. Key Design Decision: Wavelet = coif9

After systematic empirical evaluation, **Coiflet-9** was identified as the optimal wavelet for tree-sparse representations of nanopore squiggles.

**Properties of coif9:**
- Vanishing moments: ν = 18
- Filter length: 54
- Smoothness: α ≈ 3+ (high regularity)
- Orthogonal: yes (required for Ψ^T Ψ = I)

**Why coif9 is optimal for this pipeline:**

1. **Multilevel allocation efficiency:** With ν=18, fine-to-coarse interference `2^{-18·ΔR}` is essentially zero for ΔR ≥ 1. With α ≈ 3+, coarse-to-fine interference `2^{-2.5·ΔR}` is also very fast. The local coherence matrix becomes nearly perfectly block-diagonal, allowing extremely aggressive undersampling of high-frequency bands.

2. **Coiflet scaling function property:** Coiflets have vanishing moments on the scaling function too (≈ ν-1 = 17). This gives near-exact polynomial reproduction up to degree 17, which helps with smooth baseline drift in ionic current.

3. **Tree sparsity is preserved:** Despite the longer filter (54 taps), each transition still creates a connected subtree. The sparsity k increases by a constant factor per transition relative to Haar, but the multilevel efficiency gain is exponential in ν — the tradeoff strongly favors coif9.

4. **Contrast with Haar (α=1, ν=1):** Haar gives maximal raw sparsity for piecewise-constant signals but suffers from the slow `2^{-0.5·ΔR}` coarse-to-fine interference in the multilevel formula. This forces heavy sampling even at high frequencies.

---

## 4. Codebase Architecture

### 4.1 Module Map

```
src/ncs/
├── wt_coeffs.py                  # WtCoeffs container: flat ↔ grouped wavelet coefficients
├── wavelet_module.py             # Forward/inverse DWT via pywt (periodization mode, orthogonal only)
├── exact_tree_projection.py      # Cartis & Thompson (2013) DP: O(k²·n) best tree-k-sparse approx
├── reconstruction_module.py      # Model-based CoSaMP with tree projection
├── measurement_module.py         # Measurement operator factories (Gaussian, subsampling, Fourier, random modulation, wavelet packet)
├── compressed_sensing_module.py  # End-to-end pipeline: measure_and_reconstruct()
├── sparse_signal_generator.py    # Random tree-sparse signal generation + noise injection
├── wavelet_packet_module.py      # Wavelet packet best-basis selection (Coifman-Wickerhauser)
├── coherence_diagnostics.py      # [NEW, HAS BUGS] Gram matrix, mutual coherence, RIP estimation, phase transition
├── experiments.py                # Plotting experiments (MSE vs m, noisy reconstruction)
└── config.py                     # Paths (FIGURES_DIR, etc.)

tests/
├── test_wt_coeffs.py
├── test_sparse_signal_generator.py
└── ... (other test files)
```

### 4.2 Data Flow

```
generate_tree_sparse_coeffs(power, count, tree_sparsity, wavelet)
    → random signal → DWT → tree_projection → WtCoeffs (ground truth)

measure_and_reconstruct(measurement_mode, m, reconstruction_mode, coeffs_x, target_tree_sparsity)
    → creates measurement operator (Φ, Φ^T, Φ†)
    → composes with DWT/IDWT for time/frequency-domain operators
    → y = Φ(IDWT(coeffs_x))
    → CoSaMP iteration:
        e = Φ^T(r)                    # proxy
        Ω = tree_projection(e, 2k)    # identify support
        T = support(x̂) ∪ Ω           # merge
        b = least_squares(y, T)       # solve on merged support
        x̂ = tree_projection(b, k)    # prune
        r = y - Φ(x̂)                 # update residual
    → returns x̂ (WtCoeffs)
```

### 4.3 Measurement Operators Implemented

| Mode | Domain | Incoherent w/ wavelets? | RIP guarantees? | Status |
|------|--------|------------------------|-----------------|--------|
| `gaussian` | Wavelet coefficients directly | Universal | Yes, O(k log(N/k)) | ✅ Working |
| `subsampling` | Time domain | **No** (coherent) | **No** | ✅ Working (baseline) |
| `fourier_subsampling` | Frequency domain (rfft) | Yes (Candès et al.) | Yes | ✅ Working |
| `random_modulation` | Time domain (±1 chipping + subsample) | Partially | Empirical only | ✅ Working |
| `wavelet_packet` | Time domain + WP best basis | Reduced coherence via WP | Partial (WP depth dependent) | ✅ Working |
| `hadamard` | Sequency domain (WHT) | Yes (asymptotically) | Yes (multilevel) | ❌ Not implemented |
| `hadamard_multilevel` | Sequency domain (WHT, variable density) | Yes (optimally) | Yes (Thm 6.2) | ❌ Not implemented |

### 4.4 Known Bugs and WIP

**compressed_sensing_module.py:**
- Historical bug: empty tuple for unsupported measurement modes (now raises ValueError)
- Historical bug: wrong kwargs to reconstruct() (documented, not fully fixed)
- The pseudo-inverse of composed operator `(S ∘ IDWT)` is approximated as `DWT ∘ S†`, which is NOT correct. A proper least-squares pseudo-inverse may be needed for CoSaMP convergence guarantees.
- The reconstruction module materializes the full Φ matrix (m×n) for least-squares on support — this is O(mn) memory and O(mn²) for pinv. Acceptable for n ≤ 8192 but doesn't scale.

**coherence_diagnostics.py (newly created, has bugs):**
1. `phase_transition_grid`: hardcodes `measurement_mode="subsampling"` and passes unsupported `measurement_op_factory` kwarg to `measure_and_reconstruct`
2. `mutual_coherence`: normalization assumes energy-preserving operators without documenting this
3. `empirical_rip_constant`: hardcoded seed=0, should be a parameter
4. Missing: `local_coherence_matrix()` — the central quantity from Adcock et al.
5. Missing: `optimal_multilevel_allocation()` — computes per-band {m_k} from {s_l}
6. Missing: `flip_test()` — validates that operator exploits signal structure beyond sparsity

**General:**
- Signal padding: currently signals are cut to nearest power of 2 (noted in README)
- No real nanopore signal integration yet (all experiments use synthetic tree-sparse signals)

---

## 5. Completed Research

### 5.1 Deep Research Document: "Breaking the Coherence Barrier"

A comprehensive Gemini Deep Research synthesis (in project files as `Wavelet_Sampling_and_Signal_Sparsity.pdf`) covering:
- Adcock, Hansen & Poon Theorem 6.1 (asymptotic incoherence proof)
- Explicit local coherence formulas μ_{N,M}(k,l) with decay bounds
- Optimal sampling density formula (Theorem 6.2)
- Walsh-Hadamard wavelet pair: confirmed same asymptotic incoherence in dyadic blocks
- Thesing, Hansen & Antun: recovery guarantees for Walsh-wavelet
- Moshtaghpour, Bioucas-Dias & Jacques: parallel confirmation
- Model-RIP: O(C^k) subspace reduction, m = O(k) measurements
- RAmP for compressible signals
- Explicit Haar specialization of the multilevel formula (α=1, ν=1)
- Nanopore squiggle allocation analysis: slow 2^{-0.5·ΔR} penalty for Haar necessitates heavy-tailed polynomial sampling

### 5.2 Hardware-Feasible Measurement Operators Analysis

Detailed analysis of signal paths from nanopore pore current to digital measurement vector, covering:
- Subsampling approaches (plain, random modulation, multilevel filter bank, event-triggered)
- Quantization approaches (1-bit CS, sigma-delta)
- Linear transform approaches (Hadamard, random convolution, chirp modulation)
- The random demodulator architecture: x(t) → ±1 chipping → reduced-rate ADC
- Hadamard identified as highest-value path: Walsh functions are trivially hardware-implementable (square waves), asymptotically incoherent with wavelets, O(N log N) computation

### 5.3 D-AMP Analysis (Medium-Term Direction)

Denoising-based Approximate Message Passing (Metzler et al. 2016):
- Decouples measurement operator from signal prior (prior encoded in denoiser, not sparsity basis)
- Requires sub-Gaussian measurements for state evolution (Gaussian, random modulation)
- Does NOT work with structured operators (Fourier, Hadamard) without OAMP/VAMP extensions
- Tree projection could serve as the "denoiser" within D-AMP
- TV denoiser or changepoint detector may be more natural for piecewise-constant signals
- Identified as "escape hatch" from the incoherence constraint — medium-term direction

---

## 6. Reference Papers in Project

Key papers loaded in the project (with their role):

| File | Role |
|------|------|
| `Nanopore_Compressive_Sensing.pdf` | Our preprint |
| `2010_Baraniuk_modelbasedcs.pdf` | Model-based CS, Model-RIP, tree sparsity, RAmP |
| `2013_Cartis_anexacttreeprojectionalgorithmforwavelets.pdf` | Exact tree projection DP algorithm |
| `2013_Bandeira_certifyingripnphard.pdf` | RIP certification is NP-hard |
| `2016_Metzler_fromdenoisingtocompressedsensing.pdf` | D-AMP framework |
| `2012_Bach_structuredsparsityconvexoptimization.pdf` | Structured sparsity theory |
| `2015_Hegde_fastalgsforstructuredsparsity.pdf` | Fast structured sparsity algorithms |
| `2014_Hegde_afastapproxalgfortreesparserecovery.pdf` | Approximate tree projection |
| `2014_Hegde_approximationtolerantmbcs.pdf` | Approximation-tolerant model-based CS |
| `2014_Hegde_nearlylineartimembcs.pdf` | Nearly linear time model-based CS |
| `2012_hassanieh_nearlyoptimalsparsefft.pdf` | Sparse FFT |
| `2012_Laska_regimechangebitdepthcs.pdf` | Bit-depth vs measurement tradeoff |
| `1999_Baraniuk_optimaltreeapproxwithwavelets.pdf` | Optimal tree approximation |
| `2001_Cohen_treeapproximationoptimalencoding.pdf` | Tree approximation theory |
| `1994_Baraniuk_cssa.pdf` | CSSA algorithm |
| `robustuncertaintyprinciples.pdf` | Candès & Tao uncertainty principles |
| `2022_Pali_ompperformance.pdf` | OMP performance analysis |
| `2017_Huang_improvedalgsforstructuredsparserecovery.pdf` | Improved structured sparse recovery |
| `2020_Tirer_generalizingcosamptosignalsfromaunionoflowdimensionallinearsubspaces.pdf` | Generalized CoSaMP |
| `2016_Bahmani_modelsparsegradientdescent.pdf` | Model-sparse gradient descent |
| `2006_La_treebasedompalgonimageprocessing.pdf` | Tree-based OMP |
| `2011_Ophir_multiscaledictionarylearningusingwavelets.pdf` | Multiscale dictionary learning |
| `2017_Ding_jointsensingandsparsifyingdisctionaryoptimization.pdf` | Joint sensing + dictionary optimization |
| `2018_Recoskie_learningsparsewaveletrepresentations.pdf` | Learning sparse wavelet representations |
| `2018_Recoskie_gradientbasedfilterdesignfordualtreewavelettransform.pdf` | Gradient-based wavelet filter design |
| `2019_Jawali_alearningapproachforwaveletdesign.pdf` | Learning-based wavelet design |
| `2024_Frusque_robusttimeseriesdenoisingwithwaveletpackets.pdf` | Wavelet packet denoising |
| `2021_Jung_quantizedcsbyrectifiedlinearunits.pdf` | Quantized CS with ReLU |
| `2022_Dao_flashattention.pdf` | FlashAttention (context: computational efficiency) |
| `2015_Hegde_approximationalgorithmsformbcs.pdf` | Approximation algorithms for MBCS |

---

## 7. Pending Implementation — Codex Prompts

### 7.1 Fix coherence_diagnostics.py

**Priority: HIGH (prerequisite for all experiments)**

```
Task: Fix bugs and add missing features to src/ncs/coherence_diagnostics.py.

Bug fixes:
1. phase_transition_grid: Remove hardcoded measurement_mode="subsampling".
   Accept a measurement_mode: str parameter and pass it to
   measure_and_reconstruct. Remove the unsupported measurement_op_factory kwarg.

2. mutual_coherence: Add docstring note that formula assumes energy-preserving
   operators (E[‖Φx‖²] = ‖x‖²).

3. empirical_rip_constant: Make seed parameter explicit (default=0).

New features:
4. local_coherence_matrix(G, n, wavelet) -> np.ndarray:
   Partition columns of G by wavelet scale, rows into r dyadic bands.
   Compute μ_{N,M}(k,l) per (band k, scale l) pair using Adcock et al. definition:
   μ_{N,M}(k,l) = sqrt(max_{i∈band_k, j∈scale_l} |G_{ij}|² · max_{i∈band_k, j∈all} |G_{ij}|²)
   Return r × (max_level+1) matrix plus band/scale boundary metadata.

5. optimal_multilevel_allocation(local_sparsities, n, wavelet, total_m) -> np.ndarray:
   Given vector of local sparsities s_l (one per wavelet scale), compute
   per-band allocation {m_k} using Theorem 6.2 formula. Use the wavelet's
   smoothness α and vanishing moments ν from pywt. For coif9: ν=18, α≈3.2.
   If total_m provided, normalize to sum to total_m.

6. flip_test(measure_op, n, wavelet, k, n_trials=50) -> dict:
   For each trial: generate tree-k-sparse signal x, measure y = Φ·IDWT(x),
   reconstruct x_hat via CoSaMP. Then create x_flip by randomly permuting
   wavelet coefficient indices (preserving sparsity, destroying tree structure),
   measure y_flip, reconstruct x_flip_hat. Return {mse_structured, mse_flipped,
   ratio}. Ratio >> 1 means operator exploits structure.
```

### 7.2 Hadamard Measurement Operators

**Priority: HIGH (core new operator)**

```
Task: Add Walsh-Hadamard measurement operators to src/ncs/measurement_module.py.

1. create_hadamard_operator(n, m, seed=None):
   - Fast Walsh-Hadamard Transform with SEQUENCY ORDERING (not natural/Hadamard ordering).
   - Sequency = number of zero-crossings, analogous to frequency.
   - Sort rows of scipy.linalg.hadamard(n) by sequency count.
   - Subsample m rows uniformly at random, scale √(n/m).
   - Return (measure, adjoint, pseudo_inverse) triple.

2. create_hadamard_multilevel_operator(n, m, wavelet='coif9', local_sparsities=None, seed=None):
   - Partition sequency domain into r = log2(n) dyadic bands.
   - If local_sparsities provided, use Theorem 6.2 formula with wavelet's α, ν.
   - For coif9 (ν=18, α≈3.2): cross-band interference is negligible,
     allocation approximately proportional to local sparsity s_l × band size.
   - Default allocation for coif9: nearly flat (unlike Haar where heavy low-freq bias needed).
   - Within each band, subsample m_k rows uniformly at random.
   - Return (measure, adjoint, pseudo_inverse) + metadata dict with allocations.

3. Integration:
   - Register as "hadamard" and "hadamard_multilevel" in MEASUREMENT_OPERATORS.
   - Add to measure_and_reconstruct in compressed_sensing_module.py.
   - Composition: phi(wt) = S · WHT · IDWT(wt), phi_T(y) = DWT · WHT^T · S^T(y).
   - WHT is symmetric and orthogonal: WHT^T = WHT^{-1} = (1/n)·WHT.

CRITICAL: sequency ordering. Without it, multilevel sampling is meaningless.
```

### 7.3 Measurement Operator Benchmark Experiment

**Priority: MEDIUM (after operators are implemented)**

```
Task: Create src/ncs/experiments_measurement_comparison.py.

1. Generate 20 random tree-sparse signals: n=4096 (power=12), tree_sparsity=100, wavelet='coif9'.

2. For each measurement mode in ['gaussian', 'fourier_subsampling', 'random_modulation',
   'hadamard', 'hadamard_multilevel'], and m in np.linspace(150, 1500, 20).astype(int):
   - Run measure_and_reconstruct with reconstruction_mode='CoSaMP'
   - Record: measurement_mode, m, signal_index, MSE, support_recovery_rate, wall_clock_time

3. Produce figures:
   - Main: MSE vs m/k ratio, one curve per operator, with error bands
   - Inset: empirical phase transition (P(exact recovery) > 0.95) vs m/k
   - Coherence heatmaps for each operator side by side

4. Save results CSV + figures.
```

---

## 8. Pending Research — NotebookLM Prompts

### 8.1 D-AMP Compatibility with Structured Measurements

```
Analyze: D-AMP requires i.i.d. sub-Gaussian measurements for state evolution.
1. What goes wrong with structured (Hadamard, Fourier) measurements?
2. Do OAMP/VAMP extend guarantees to these?
3. Can tree-sparse projection serve as the "denoiser" in D-AMP?
4. For 1D piecewise-constant signals, is TV denoiser more natural than tree thresholding?
```

### 8.2 1-Bit and Low-Bit CS with Tree Sparsity

```
1. Can BIHT be extended with tree-sparse projection?
2. Laska et al. 2012: what is the optimal (m, B) operating point for nanopore signals?
3. Does the model-based advantage persist under quantization?
```

---

## 9. Strategic Roadmap

### Immediate (implements → paper figures)
1. Fix coherence_diagnostics.py
2. Implement Hadamard operators (uniform + multilevel)
3. Run flip test: if ratio >> 1 for multilevel Hadamard + coif9 but ≈ 1 for Gaussian → paper figure proving structure exploitation
4. Coherence heatmap comparison across all operators → paper figure showing block-diagonal structure
5. Phase transition benchmark across operators → core results figure

### Medium-term (extends paper scope)
6. D-AMP with tree projection as denoiser
7. 1-bit / low-bit CS with tree sparsity
8. Real nanopore signal integration (replace synthetic signals)
9. Wavelet packet best-basis vs fixed coif9 comparison

### Paper contributions
- **Empirical:** First application of multilevel Hadamard sampling with tree-sparse CoSaMP to nanopore signals
- **Theoretical insight (unpublished gap):** Tree-sparse local sparsity profiles {s_l} fed into multilevel allocation give dramatically better efficiency than plain sparsity assumptions
- **Engineering:** Concrete hardware-feasible measurement operator (WHT = square waves) with quantified compression ratios

---

## 10. Tools and Workflow

- **NotebookLM:** RAG-augmented research with full project context. Use for theoretical questions, literature synthesis, formula derivation.
- **Codex:** Code generation and implementation with full repo context. Use for new modules, bug fixes, experiment scripts.
- **Claude (orchestrator):** Strategic coordination, prompt generation, architectural decisions, cross-cutting analysis.

**Constraint:** The researcher is only marginally interested in proofs — only to the degree they help with empirical results. Prioritize operational clarity and falsifiable experiments.

---

## 11. Environment

- Python with PyWavelets (pywt), NumPy, SciPy, Matplotlib, Pandas, Seaborn, tqdm
- Package manager: uv
- Formatter: ruff
- Signals are padded/cut to power-of-2 lengths
- All wavelets must be orthogonal (Ψ^T Ψ = I requirement)
- Periodization mode for DWT (exact n-dimensional isomorphism)
