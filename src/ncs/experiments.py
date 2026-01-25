import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from ncs.compressed_sensing_module import measure_and_reconstruct
from ncs.sparse_signal_generator import generate_tree_sparse_coeffs, add_noise_to_coeffs
from ncs.wavelet_module import inverse_transform
import os


def _worker_for_m(
    m, coeff_pairs, measurement_mode, reconstruction_mode, target_sparsity, attempts
):
    """
    Worker function to process a single value of m (measurements).
    It runs reconstructions for all provided coefficient pairs.
    """
    batch_results = []

    # coeff_pairs is a list of tuples: (clean_ground_truth, input_for_measurement)
    for clean_coeff, input_coeff in coeff_pairs:
        for _ in range(attempts):
            x_hat = measure_and_reconstruct(
                measurement_mode=measurement_mode,
                m=int(m),
                reconstruction_mode=reconstruction_mode,
                coeffs_x=input_coeff,
                target_tree_sparsity=target_sparsity,
            )

            # Calculate metrics
            sparse_z = inverse_transform(clean_coeff)
            reconstructed_z = inverse_transform(x_hat)

            missed_support_len = len(clean_coeff.support - x_hat.support)
            mse = np.mean((sparse_z - reconstructed_z) ** 2)

            batch_results.append(
                {"m": m, "mse": mse, "missed_support": missed_support_len}
            )
    return batch_results


def _run_parallel_experiment(
    m_values,
    coeff_pairs,
    measurement_mode,
    reconstruction_mode,
    tree_sparsity,
    attempts,
    max_workers=os.cpu_count(),
):
    """
    Orchestrates the parallel execution over m_values.
    """
    results = []

    # Use ThreadPoolExecutor for threading.
    # Note: If your tasks are very CPU heavy in pure Python, consider ProcessPoolExecutor instead.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_m = {
            executor.submit(
                _worker_for_m,
                m,
                coeff_pairs,
                measurement_mode,
                reconstruction_mode,
                tree_sparsity,
                attempts,
            ): m
            for m in m_values
        }

        # Gather results with a progress bar
        for future in tqdm(
            as_completed(future_to_m),
            total=len(m_values),
            desc="Parallel Reconstruction",
        ):
            try:
                data = future.result()
                results.extend(data)
            except Exception as exc:
                m_val = future_to_m[future]
                print(f"Measurement m={m_val} generated an exception: {exc}")

    return results


def generate_random_sparse_signal_reconstruction_data(
    n_power,
    tree_sparsity,
    wavelet,
    measurement_mode,
    reconstruction_mode,
    signal_count,
    reconstruction_attempts,
    m_values,
    max_workers=os.cpu_count(),
):
    sparse_coeffs = generate_tree_sparse_coeffs(
        power=n_power,
        count=signal_count,
        tree_sparsity=tree_sparsity,
        wavelet=wavelet,
    )

    # For perfect reconstruction, input is same as ground truth
    # Create pairs of (clean, clean)
    coeff_pairs = [(c, c) for c in sparse_coeffs]

    results = _run_parallel_experiment(
        m_values,
        coeff_pairs,
        measurement_mode,
        reconstruction_mode,
        tree_sparsity,
        reconstruction_attempts,
        max_workers,
    )

    _plot_results(
        results,
        n_power,
        tree_sparsity,
        wavelet,
        measurement_mode,
        reconstruction_mode,
        signal_count,
        reconstruction_attempts,
        m_values,
        title="Perfectly tree-sparse signal reconstruction: MSE vs Number of Measurements",
    )


def plot_noisy_signal_reconstruction_data(
    n_power,
    tree_sparsity,
    noise_epsilon,
    noise_mode,
    wavelet,
    measurement_mode,
    reconstruction_mode,
    signal_count,
    reconstruction_attempts,
    m_values,
    max_workers=os.cpu_count(),
):
    sparse_coeffs = generate_tree_sparse_coeffs(
        power=n_power,
        count=signal_count,
        tree_sparsity=tree_sparsity,
        wavelet=wavelet,
    )
    noisy_coeffs = add_noise_to_coeffs(sparse_coeffs, noise_epsilon, noise_mode)

    # Create pairs of (clean, noisy)
    coeff_pairs = list(zip(sparse_coeffs, noisy_coeffs))

    results = _run_parallel_experiment(
        m_values,
        coeff_pairs,
        measurement_mode,
        reconstruction_mode,
        tree_sparsity,
        reconstruction_attempts,
        max_workers,
    )

    extra_params = {"Noise-Epsilon": noise_epsilon, "Noise-Mode": noise_mode}

    _plot_results(
        results,
        n_power,
        tree_sparsity,
        wavelet,
        measurement_mode,
        reconstruction_mode,
        signal_count,
        reconstruction_attempts,
        m_values,
        title="Noisy tree-sparse signal reconstruction: MSE vs Number of Measurements",
        extra_params=extra_params,
    )


def _plot_results(
    results,
    n_power,
    tree_sparsity,
    wavelet,
    measurement_mode,
    reconstruction_mode,
    signal_count,
    reconstruction_attempts,
    m_values,
    title,
    extra_params=None,
):
    df = pd.DataFrame(results)
    n = 2**n_power

    params = {
        "N": n,
        "Tree-sparsity": tree_sparsity,
        "Wavelet": wavelet,
        "Measurement": measurement_mode,
        "Reconstruction": reconstruction_mode,
        "Signals": signal_count,
        "Reconstructions per signal": reconstruction_attempts,
    }
    if extra_params:
        params.update(extra_params)

    param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.stripplot(
        data=df,
        x="m",
        y="mse",
        hue="missed_support",
        jitter=0.2,
        alpha=0.7,
        palette="RdYlGn_r",
        native_scale=True,
        ax=ax,
    )

    sns.lineplot(
        data=df,
        x="m",
        y="mse",
        estimator="mean",
        color="blue",
        linewidth=2,
        errorbar=None,
        ax=ax,
        label="Mean MSE",
    )

    ax.set_xlabel(
        f"Number of Measurements (m {m_values.min()} - {m_values.max()})", fontsize=12
    )
    ax.set_ylabel("Signal domain MSE", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    ax.legend(title="Missed Support", fontsize=10, loc="upper right")

    at = AnchoredText(
        s=param_str,
        loc="upper right",
        bbox_to_anchor=(1, 0.73),
        bbox_transform=ax.transAxes,
        frameon=True,
    )
    ax.add_artist(at)

    plt.tight_layout()
    plt.show()
