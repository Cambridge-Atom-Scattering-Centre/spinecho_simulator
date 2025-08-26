from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim.molecule import build_diatomic_hamiltonian_majorana
from spinecho_sim.molecule.hamiltonian_majorana import (
    build_quadrupole_block_majorana,
    build_spin_rotational_hamiltonian_majorana,
    build_zeeman_hamiltonian_majorana,
)
from spinecho_sim.util import csr_to_array, plot_complex_heatmap

if __name__ == "__main__":
    H = 300
    a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
    test_result = csr_to_array(
        build_zeeman_hamiltonian_majorana(n_i=2, n_j=2, a=a, b=b, b_vec=(0.2, 0.2, 1.0))
    )
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(r"Linear Field Terms $-a\mathbf{I \cdot H} - b\mathbf{J \cdot H}$")

    test_result = csr_to_array(
        build_spin_rotational_hamiltonian_majorana(n_i=2, n_j=2, c=c)
    )
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(r"Spin Rotational Term $-c\mathbf{I \cdot J}$")

    test_result = csr_to_array(build_quadrupole_block_majorana(n_i=2, n_j=2, d=d))
    fig, ax = plot_complex_heatmap(test_result)
    ax.set_title(
        r"Quadrupole Term $+\frac{5d}{(2J-1)(2J+3)} [3(\mathbf{I \cdot J})^2 + \frac{3}{2} \mathbf{I \cdot J} -\mathbf{I}^2 \mathbf{J}^2]$"
    )

    test_result = build_diatomic_hamiltonian_majorana(
        n_i=2, n_j=2, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
    )
    test_result_array = csr_to_array(test_result)
    print(test_result.nnz)
    print(
        "sparsity:",
        f"{test_result.nnz} nnz elements,",
        f"{test_result.nnz / test_result_array.size:.1%}",
    )
    fig, ax = plot_complex_heatmap(test_result_array)

    plt.show()
