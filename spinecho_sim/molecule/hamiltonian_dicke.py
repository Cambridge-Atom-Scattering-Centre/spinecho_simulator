from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp  # type: ignore[import]

from spinecho_sim.util import plot_complex_heatmap, sparse_matmul, to_array


def single_spin_ops_sparse(
    s: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    dim = int(2 * s + 1)
    mj_values = np.arange(s, -s - 1, -1)
    j_z = sp.diags(mj_values, 0, format="csr")

    # ladder matrices
    j_plus = sp.lil_matrix((dim, dim), dtype=complex)
    for m in range(dim - 1):
        j_plus[m, m + 1] = np.sqrt(
            s * (s + 1) - mj_values[m + 1] * (mj_values[m + 1] - 1)
        )
    j_plus = j_plus.tocsr()
    j_minus = j_plus.transpose().conj()

    j_x = 0.5 * (j_plus + j_minus)
    j_y = -0.5j * (j_plus - j_minus)
    return j_x, j_y, j_z


def collective_ops_sparse(
    i: float, j: float
) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    i_x, i_y, i_z = single_spin_ops_sparse(i)
    j_x, j_y, j_z = single_spin_ops_sparse(j)

    dim_i, dim_j = i_x.shape[0], j_x.shape[0]
    operators_i = [sp.kron(op, sp.eye(dim_j, format="csr")) for op in (i_x, i_y, i_z)]
    operators_j = [sp.kron(sp.eye(dim_i, format="csr"), op) for op in (j_x, j_y, j_z)]
    return operators_i, operators_j


def ramsey_sparse(
    i: float,
    j: float,
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    a, b, c, d = coefficients
    i_ops, j_ops = collective_ops_sparse(i, j)
    i_x, i_y, i_z, j_x, j_y, j_z = (*i_ops, *j_ops)
    b_x, b_y, b_z = b_vec
    b_mag = np.linalg.norm(b_vec)

    # 1) linear Zeeman
    hamiltonian_i = -a * (b_x * i_x + b_y * i_y + b_z * i_z) / b_mag
    hamiltonian_j = -b * (b_x * j_x + b_y * j_y + b_z * j_z) / b_mag

    # 2) spin–rotation
    i_dot_j = (
        sparse_matmul(i_x, j_x) + sparse_matmul(i_y, j_y) + sparse_matmul(i_z, j_z)
    )
    hamiltonian_spin_rotation = -c * i_dot_j

    # 3) quadrupole / spin–spin
    i_sq = sparse_matmul(i_x, i_x) + sparse_matmul(i_y, i_y) + sparse_matmul(i_z, i_z)
    j_sq = sparse_matmul(j_x, j_x) + sparse_matmul(j_y, j_y) + sparse_matmul(j_z, j_z)
    hamiltonian_quadrupole = (5 * d / ((2 * j - 1) * (2 * j + 3))) * (
        3 * sparse_matmul(i_dot_j, i_dot_j) + 1.5 * i_dot_j - sparse_matmul(i_sq, j_sq)
    )

    return (
        hamiltonian_i
        + hamiltonian_j
        + hamiltonian_spin_rotation
        + hamiltonian_quadrupole
    )


# result = single_spin_ops(1.0)
# fig, ax = plot_complex_heatmap(result[0])
# fig, ax = plot_complex_heatmap(result[1])
# fig, ax = plot_complex_heatmap(result[2])

# i_ops, j_ops = collective_ops(i=1.0, j=1.0)
# i_x, i_y, i_z, j_x, j_y, j_z = (*i_ops, *j_ops)  # unpack
# fig, ax = plot_complex_heatmap(i_x)
# fig, ax = plot_complex_heatmap(i_y)
# fig, ax = plot_complex_heatmap(i_z)
# fig, ax = plot_complex_heatmap(j_x)
# fig, ax = plot_complex_heatmap(j_y)
# fig, ax = plot_complex_heatmap(j_z)

# Example usage:
H = 300
a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
full_result_sparse = ramsey_sparse(
    1, 1, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
)
full_result_array = to_array(full_result_sparse)
print(
    "sparsity:",
    f"{full_result_sparse.nnz} nnz elements,",
    f"{full_result_sparse.nnz / (full_result_array.size):.1%}",
)
fig, ax = plot_complex_heatmap(full_result_array)
plt.show()
