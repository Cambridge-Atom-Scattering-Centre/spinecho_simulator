from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from spinecho_sim.molecule import (
    diatomic_hamiltonian_dicke,
    diatomic_hamiltonian_majorana,
)
from spinecho_sim.util import to_array

if TYPE_CHECKING:
    import scipy.sparse as sp  # pyright: ignore[reportMissingTypeStubs]

# Original colormap
original_cmap = plt.get_cmap("bwr")
# Crop to the first half
cropped_cmap = LinearSegmentedColormap.from_list(
    "cropped_bwr", original_cmap(np.linspace(0.5, 1, 256))
)


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
    ramsey: tuple[int, int, float]
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

            # Ramsey Hamiltonian
            ramsey_hamiltonian = diatomic_hamiltonian_dicke(
                i=i, j=j, coefficients=coefficients, b_vec=b_vec
            )
            ramsey_metrics = compute_sparsity(ramsey_hamiltonian)

            # Majorana Hamiltonian
            majorana_hamiltonian = diatomic_hamiltonian_majorana(
                n_i=int(2 * i), n_j=int(2 * j), coefficients=coefficients, b_vec=b_vec
            )
            majorana_metrics = compute_sparsity(majorana_hamiltonian)

            # Store results in the dataclass
            results.append(
                Result(i=i, j=j, ramsey=ramsey_metrics, majorana=majorana_metrics)
            )

    return results


def generate_heatmaps(
    results: list[Result],
    i_values: list[float],
    j_values: list[float],
) -> None:
    """Generate heatmaps for differences in nnz, size, and sparsity."""
    nnz_diff = np.zeros((len(i_values), len(j_values)))
    size_diff = np.zeros((len(i_values), len(j_values)))
    sparsity_diff = np.zeros((len(i_values), len(j_values)))

    for result in results:
        i_idx = i_values.index(result.i)
        j_idx = j_values.index(result.j)

        nnz_diff[i_idx, j_idx] = result.majorana[0] - result.ramsey[0]
        size_diff[i_idx, j_idx] = result.majorana[1] - result.ramsey[1]
        sparsity_diff[i_idx, j_idx] = result.majorana[2] - result.ramsey[2]

    # Plot heatmaps (unchanged)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(
        np.log(nnz_diff), cmap=cropped_cmap, origin="lower", aspect="auto"
    )
    axes[0].set_title("NNZ Difference (Majorana - Ramsey)")
    axes[0].set_xlabel("J Values")
    axes[0].set_ylabel("I Values")
    axes[0].set_xticks(range(len(j_values)))
    axes[0].set_xticklabels(j_values)
    axes[0].set_yticks(range(len(i_values)))
    axes[0].set_yticklabels(i_values)
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        np.log(size_diff), cmap=cropped_cmap, origin="lower", aspect="auto"
    )
    axes[1].set_title("Size Difference (Majorana - Ramsey)")
    axes[1].set_xlabel("J Values")
    axes[1].set_ylabel("I Values")
    axes[1].set_xticks(range(len(j_values)))
    axes[1].set_xticklabels(j_values)
    axes[1].set_yticks(range(len(i_values)))
    axes[1].set_yticklabels(i_values)
    fig.colorbar(im2, ax=axes[1])

    norm_sparsity = TwoSlopeNorm(
        vmin=sparsity_diff.min(), vcenter=0, vmax=sparsity_diff.max()
    )
    im3 = axes[2].imshow(
        sparsity_diff,
        cmap="bwr",
        norm=norm_sparsity,
        origin="lower",
        aspect="auto",
    )
    axes[2].set_title("Sparsity Difference (Majorana - Ramsey)")
    axes[2].set_xlabel("J Values")
    axes[2].set_ylabel("I Values")
    axes[2].set_xticks(range(len(j_values)))
    axes[2].set_xticklabels(j_values)
    axes[2].set_yticks(range(len(i_values)))
    axes[2].set_yticklabels(i_values)
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
            f"  Ramsey: nnz={result.ramsey[0]}, size={result.ramsey[1]}, sparsity={result.ramsey[2]:.1%}"
        )
        print(
            f"  Majorana: nnz={result.majorana[0]}, size={result.majorana[1]}, sparsity={result.majorana[2]:.1%}"
        )

    # Generate heatmaps
    generate_heatmaps(results, i_values, j_values)
