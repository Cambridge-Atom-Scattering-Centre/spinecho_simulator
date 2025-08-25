from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from spinecho_sim.molecule import (
    diatomic_hamiltonian_dicke,
    diatomic_hamiltonian_majorana,
)
from spinecho_sim.util import to_array

if TYPE_CHECKING:
    import scipy.sparse as sp  # pyright: ignore[reportMissingTypeStubs]


def compute_sparsity(hamiltonian: sp.csr_matrix) -> tuple[int, int, float]:
    """Compute sparsity metrics for a given Hamiltonian."""
    hamiltonian_array = to_array(hamiltonian)
    nnz = hamiltonian.nnz
    size = hamiltonian_array.size
    sparsity = nnz / size
    return nnz, size, sparsity


@dataclass
class Result:
    """Dataclass to store sparsity comparison results."""

    i: float
    j: float
    dicke: tuple[int, int, float]
    majorana: tuple[int, int, float]


def compare_sparsity(
    i_values: list[float],
    j_values: list[float],
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> list[Result]:
    """Compare sparsity metrics for various i and j values."""
    results: list[Result] = []  # Store results as a list of dataclass instances

    for i in i_values:
        for j in j_values:
            print(f"Computing for I={i}, J={j}...")

            # Dicke Hamiltonian
            dicke_hamiltonian = diatomic_hamiltonian_dicke(
                i=i, j=j, coefficients=coefficients, b_vec=np.array(b_vec)
            )
            dicke_metrics = compute_sparsity(dicke_hamiltonian)

            # Majorana Hamiltonian
            majorana_hamiltonian = diatomic_hamiltonian_majorana(
                n_i=int(2 * i), n_j=int(2 * j), coefficients=coefficients, b_vec=b_vec
            )
            majorana_metrics = compute_sparsity(majorana_hamiltonian)

            # Store results in the dataclass
            results.append(
                Result(i=i, j=j, dicke=dicke_metrics, majorana=majorana_metrics)
            )

    return results


def calculate_differences(
    results: list[Result],
    i_values: list[float],
    j_values: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate differences in nnz, size, and sparsity."""
    nnz_diff = np.zeros((len(i_values), len(j_values)))
    size_diff = np.zeros((len(i_values), len(j_values)))
    sparsity_diff = np.zeros((len(i_values), len(j_values)))

    for result in results:
        i_idx = i_values.index(result.i)
        j_idx = j_values.index(result.j)

        nnz_diff[i_idx, j_idx] = result.majorana[0] - result.dicke[0]
        size_diff[i_idx, j_idx] = result.majorana[1] - result.dicke[1]
        sparsity_diff[i_idx, j_idx] = result.majorana[2] - result.dicke[2]

    return nnz_diff, size_diff, sparsity_diff


def generate_heatmaps(  # noqa: PLR0914
    results: list[Result],
    i_values: list[float],
    j_values: list[float],
    normalize_to: str = "lowest",  # "lowest" or "highest"
) -> None:
    """Generate heatmaps for differences in nnz, size, and sparsity."""
    nnz_diff, size_diff, sparsity_diff = calculate_differences(
        results, i_values, j_values
    )

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax in axes:
        ax.set_xlabel("J Values")
        ax.set_ylabel("I Values")
        ax.set_xticks(range(len(j_values)))
        ax.set_xticklabels(j_values)
        ax.set_yticks(range(len(i_values)))
        ax.set_yticklabels(i_values)

    # Normalize NNZ Difference Heatmap
    if normalize_to == "lowest":
        vmin_nnz = 0
        vmax_nnz = nnz_diff.max()
    else:  # normalize_to == "highest"
        vmin_nnz = nnz_diff.min()
        vmax_nnz = 0
    norm_nnz = Normalize(vmin=vmin_nnz, vmax=vmax_nnz)
    im1 = axes[0].imshow(nnz_diff, norm=norm_nnz, origin="lower", aspect="auto")
    axes[0].set_title("NNZ Difference (Majorana - Dicke)")
    fig.colorbar(im1, ax=axes[0])

    # Normalize Size Difference Heatmap
    if normalize_to == "lowest":
        vmin_size = 0
        vmax_size = size_diff.max()
    else:  # normalize_to == "highest"
        vmin_size = size_diff.min()
        vmax_size = 0
    norm_size = Normalize(vmin=vmin_size, vmax=vmax_size)
    im2 = axes[1].imshow(size_diff, norm=norm_size, origin="lower", aspect="auto")
    axes[1].set_title("Size Difference (Majorana - Dicke)")
    fig.colorbar(im2, ax=axes[1])

    # Normalize Sparsity Difference Heatmap
    if normalize_to == "lowest":
        vmin_sparsity = 0
        vmax_sparsity = sparsity_diff.max()
    else:  # normalize_to == "highest"
        vmin_sparsity = sparsity_diff.min()
        vmax_sparsity = 0
    norm_sparsity = Normalize(vmin=vmin_sparsity, vmax=vmax_sparsity)
    im3 = axes[2].imshow(
        sparsity_diff, norm=norm_sparsity, origin="lower", aspect="auto"
    )
    axes[2].set_title("Sparsity Difference (Majorana - Dicke)")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Coefficients and magnetic field vector
    H = 300
    a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
    b_vec = (0.2, 0.2, 1.0)

    # Values for I and J
    i_values = [float(round(x, 1)) for x in np.arange(1, 3, 0.5)]
    j_values = [float(round(x, 1)) for x in np.arange(1, 3, 0.5)]

    # Compare sparsity
    results = compare_sparsity(
        i_values, j_values, coefficients=(a, b, c, d), b_vec=b_vec
    )

    # Display results
    for result in results:
        print(f"I={result.i}, J={result.j}")
        print(
            f"  Dicke: nnz={result.dicke[0]}, size={result.dicke[1]}, sparsity={result.dicke[2]:.1%}"
        )
        print(
            f"  Majorana: nnz={result.majorana[0]}, size={result.majorana[1]}, sparsity={result.majorana[2]:.1%}"
        )

    # Generate heatmaps
    generate_heatmaps(results, i_values, j_values)
