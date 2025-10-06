from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.molecule import build_diatomic_hamiltonian_dicke
from spinecho_sim.molecule.hamiltonian_dicke import (
    build_collective_operators,
    build_single_spin_operators,
)
from spinecho_sim.util import csr_to_array, plot_complex_heatmap

# This example demonstrates the creation of operators for the diatomic Hamiltonian
# in the Dicke angular momentum basis. It then visualizes the magnitude and phase
# of the matrix elements of the Hamiltonian components and the relevant operators
# used in its construction through heatmaps.

if __name__ == "__main__":
    # Generate single-spin angular momentum operators (Jx, Jy, Jz) in sparse form
    operators = build_single_spin_operators(two_s=2)

    # Plot heatmaps for the single-spin operators
    fig, ax = plot_complex_heatmap(csr_to_array(operators[0]))
    ax.set_title(r"$J_x$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(operators[1]))
    ax.set_title(r"$J_y$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(operators[2]))
    ax.set_title(r"$J_z$, $J=1$")

    # Generate collective angular momentum operators (Ix, Iy, Iz, Jx, Jy, Jz) in sparse form
    i_ops, j_ops = build_collective_operators(two_i=2, two_j=2)
    i_x, i_y, i_z, j_x, j_y, j_z = (*i_ops, *j_ops)  # Unpack the operators

    # Plot heatmaps for the collective operators
    fig, ax = plot_complex_heatmap(csr_to_array(i_x))
    ax.set_title(r"$I_x \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(i_y))
    ax.set_title(r"$I_y \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(i_z))
    ax.set_title(r"$I_z \otimes 1$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(j_x))
    ax.set_title(r"$1 \otimes J_x$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(j_y))
    ax.set_title(r"$1 \otimes J_y$, $I=1$, $J=1$")
    fig, ax = plot_complex_heatmap(csr_to_array(j_z))
    ax.set_title(r"$1 \otimes J_z$, $I=1$, $J=1$")

    # Construct the full diatomic Hamiltonian in the Dicke angular momentum basis
    full_result_sparse = build_diatomic_hamiltonian_dicke(
        two_i=2,  # Spin quantum number I
        two_j=2,  # Spin quantum number J
        coefficients=(
            2 * np.pi * 4.258e7,  # Coefficient for I·H interaction
            2 * np.pi * 0.66717e7,  # Coefficient for J·H interaction
            2 * np.pi * 113.8e3,  # Coefficient for I·J interaction
            2 * np.pi * 57.68e3,  # Coefficient for quadrupole interaction
        ),
        b_vec=np.array([0.2, 0.2, 1.0]),  # Magnetic field vector
    )

    # Convert the sparse Hamiltonian to a dense array for visualization
    full_result_array = csr_to_array(full_result_sparse)

    # Plot a heatmap for the full Hamiltonian
    fig, ax = plot_complex_heatmap(full_result_array)
    ax.set_title(
        r"$H$, $I=1$, $J=1$, "
        f"{full_result_sparse.nnz} nnz elements, "  # Number of non-zero elements
        f"{full_result_sparse.nnz / (full_result_array.size):.1%}"  # Sparsity percentage
    )

    # Display all the plots
    plt.show()
