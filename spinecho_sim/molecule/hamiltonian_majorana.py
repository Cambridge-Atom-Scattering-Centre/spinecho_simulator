from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp  # type: ignore[import]

from spinecho_sim.util import kronecker_n, plot_complex_heatmap, sparse_matmul, to_array

sigma_x = sp.csr_matrix([[0, 1], [1, 0]])
sigma_y = sp.csr_matrix([[0, -1j], [1j, 0]])
sigma_z = sp.csr_matrix([[1, 0], [0, -1]])
identity = sp.csr_matrix([[1, 0], [0, 1]])


def zeeman_hamiltonian_majorana(
    *,
    n_i: int,
    n_j: int,
    a: float,
    b: float,
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Construct the Zeeman Hamiltonian for two spin systems."""
    dim = 2 ** (n_i + n_j)
    hamiltonian = sp.csr_matrix((dim, dim), dtype=complex)
    b_mag = np.linalg.norm(b_vec)

    # Terms for I subsystem
    for k in range(n_i):
        operator_list = [identity] * (n_i + n_j)
        for component, operator_i in zip(
            b_vec, (sigma_x, sigma_y, sigma_z), strict=True
        ):
            operator_list[k] = operator_i
            hamiltonian -= a * component * kronecker_n(operator_list) / b_mag
            operator_list[k] = identity

    # Terms for J subsystem
    for l in range(n_i, n_i + n_j):
        operator_list = [identity] * (n_i + n_j)
        for component, operator_j in zip(
            b_vec, (sigma_x, sigma_y, sigma_z), strict=True
        ):
            operator_list[l] = operator_j
            hamiltonian -= b * component * kronecker_n(operator_list) / b_mag
            operator_list[l] = identity

    return hamiltonian  # type: ignore[return-value]


def spin_rotational_block_majorana(*, n_i: int, n_j: int, c: float) -> sp.csr_matrix:
    """Construct the spin rotational block for two spin systems."""
    hamiltonian = sp.csr_matrix((2 ** (n_i + n_j),) * 2, dtype=complex)
    operator_list = [identity] * (n_i + n_j)
    for k in range(n_i):
        for l in range(n_i, n_i + n_j):
            for operator_i, operator_j in zip(
                (sigma_x, sigma_y, sigma_z), (sigma_x, sigma_y, sigma_z), strict=True
            ):
                operator_list[k] = operator_i
                operator_list[l] = operator_j
                hamiltonian -= c * kronecker_n(operator_list)
                operator_list[k] = identity
                operator_list[l] = identity
    return hamiltonian  # type: ignore[return-value]


def collective_ij_majorana(
    n_i: int, n_j: int
) -> tuple[dict[str, sp.csr_matrix], dict[str, sp.csr_matrix]]:
    """Return dicts of collective operators Iα, Jα (sparse CSR)."""
    dim = 2 ** (n_i + n_j)
    i_alpha: dict[str, sp.csr_matrix] = {}
    j_alpha: dict[str, sp.csr_matrix] = {}

    # build Iα ---------------------------------------------------------------
    for component, operator in zip(
        ("x", "y", "z"), (sigma_x, sigma_y, sigma_z), strict=True
    ):
        operator_sum = sp.csr_matrix((dim, dim), dtype=complex)
        for k in range(n_i):
            operator_list = [identity] * (n_i + n_j)
            operator_list[k] = operator
            operator_sum += kronecker_n(operator_list)  # factor 1/2 in definition
        i_alpha[component] = operator_sum

    # build Jα ---------------------------------------------------------------
    for component, operator in zip(
        ("x", "y", "z"), (sigma_x, sigma_y, sigma_z), strict=True
    ):
        operator_sum = sp.csr_matrix((dim, dim), dtype=complex)
        for l in range(n_j):
            operator_list = [identity] * (n_i + n_j)
            operator_list[n_i + l] = operator  # rotational qubits start at N_I
            operator_sum += kronecker_n(operator_list)
        j_alpha[component] = operator_sum

    return i_alpha, j_alpha


def quadrupole_block_majorana(n_i: int, n_j: int, d: float) -> sp.csr_matrix:
    """Return the d-term CSR matrix for given (N_I,N_J)."""
    i_alpha, j_alpha = collective_ij_majorana(n_i, n_j)

    # I⋅J -------------------------------------------------------------------
    i_dot_j = (
        sparse_matmul(i_alpha["x"], j_alpha["x"])
        + sparse_matmul(i_alpha["y"], j_alpha["y"])
        + sparse_matmul(i_alpha["z"], j_alpha["z"])
    )

    # (I⋅J)^2 ---------------------------------------------------------------
    ij_sq = sparse_matmul(i_dot_j, i_dot_j)  # sparse-sparse matmul → CSR

    # I² and J² -----------------------------------------------------------
    i_sq = (
        sparse_matmul(i_alpha["x"], i_alpha["x"])
        + sparse_matmul(i_alpha["y"], i_alpha["y"])
        + sparse_matmul(i_alpha["z"], i_alpha["z"])
    )
    j_sq = (
        sparse_matmul(j_alpha["x"], j_alpha["x"])
        + sparse_matmul(j_alpha["y"], j_alpha["y"])
        + sparse_matmul(j_alpha["z"], j_alpha["z"])
    )

    hamiltonian = (5 * d / ((n_j - 1) * (n_j + 3))) * (
        3.0 * ij_sq + 1.5 * i_dot_j - sparse_matmul(i_sq, j_sq)
    )
    hamiltonian.eliminate_zeros()  # keep it neat
    return hamiltonian


def diatomic_hamiltonian_majorana(
    n_i: int,
    n_j: int,
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Construct the diatomic Hamiltonian for two spin systems."""
    a, b, c, d = coefficients
    zeeman = zeeman_hamiltonian_majorana(n_i=n_i, n_j=n_j, a=a, b=b, b_vec=b_vec)
    rotational = spin_rotational_block_majorana(n_i=n_i, n_j=n_j, c=c)
    quadrupole = quadrupole_block_majorana(n_i=n_i, n_j=n_j, d=d)
    return zeeman + rotational + quadrupole  # type: ignore[return-value]


H = 300
a, b, c, d = 4.258 * H, 0.6717 * H, 113.8, 57.68
# test_result = to_array(
#     zeeman_hamiltonian(n_i=2, n_j=2, a=a, b=b, b_vec=(0.2, 0.2, 1.0))
# )
# fig, ax = plot_complex_heatmap(test_result)
# ax.set_title(r"Linear Field Terms $-a\mathbf{I \cdot H} - b\mathbf{J \cdot H}$")

# test_result = to_array(spin_rotational_block(n_i=2, n_j=2, c=c))
# fig, ax = plot_complex_heatmap(test_result)
# ax.set_title(r"Spin Rotational Term $-c\mathbf{I \cdot J}$")

# test_result = to_array(quadrupole_block(n_i=2, n_j=2, d=d))
# fig, ax = plot_complex_heatmap(test_result)
# ax.set_title(
#     r"Quadrupole Term $+\frac{5d}{(2J-1)(2J+3)} [3(\mathbf{I \cdot J})^2 + \frac{3}{2} \mathbf{I \cdot J} -\mathbf{I}^2 \mathbf{J}^2]$"
# )

test_result = diatomic_hamiltonian_majorana(
    n_i=2, n_j=2, coefficients=(a, b, c, d), b_vec=(0.2, 0.2, 1.0)
)
test_result_array = to_array(test_result)
print(test_result.nnz)
print(
    "sparsity:",
    f"{test_result.nnz} nnz elements,",
    f"{test_result.nnz / test_result_array.size:.1%}",
)
fig, ax = plot_complex_heatmap(test_result_array)

plt.show()
