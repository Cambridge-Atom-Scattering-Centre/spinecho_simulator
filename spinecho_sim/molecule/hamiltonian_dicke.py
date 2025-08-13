from __future__ import annotations

from functools import reduce

import numpy as np
import scipy.sparse as sp  # type: ignore[import]

from spinecho_sim.util import (
    csr_add,
    csr_diags,
    csr_eye,
    csr_hermitian,
    csr_kron,
    csr_scale,
    csr_subtract,
    sparse_matmul,
)


def single_spin_ops_sparse(
    s: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generate sparse matrices for single spin operators Sx, Sy, Sz."""
    dim = int(2 * s + 1)
    mj_values = np.arange(s, -s - 1, -1)
    j_z = csr_diags(mj_values, 0)

    # ladder matrices
    j_plus_lil = sp.lil_matrix((dim, dim), dtype=complex)
    for m in range(dim - 1):
        j_plus_lil[m, m + 1] = np.sqrt(
            s * (s + 1) - mj_values[m + 1] * (mj_values[m + 1] - 1)
        )
    j_plus = sp.csr_matrix(j_plus_lil.tocsr())
    j_minus = csr_hermitian(j_plus)

    j_x = csr_scale(csr_add(j_plus, j_minus), 0.5)
    j_y = csr_scale(csr_subtract(j_plus, j_minus), -0.5j)
    return j_x, j_y, j_z


def collective_ops_sparse(
    i: float, j: float
) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    """Generate sparse matrices for single spin operators acting on the space of two spins."""
    i_x, i_y, i_z = single_spin_ops_sparse(i)
    j_x, j_y, j_z = single_spin_ops_sparse(j)

    dim_i, dim_j = 2 * i + 1, 2 * j + 1
    identity_i = csr_eye(dim_i)
    identity_j = csr_eye(dim_j)

    operators_i = [csr_kron(op, identity_j) for op in (i_x, i_y, i_z)]
    operators_j = [csr_kron(identity_i, op) for op in (j_x, j_y, j_z)]
    return operators_i, operators_j


def diatomic_hamiltonian_dicke(
    i: float,
    j: float,
    coefficients: tuple[float, float, float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Generate the Ramsey Hamiltonian as a sparse matrix."""
    a, b, c, d = coefficients
    i_ops, j_ops = collective_ops_sparse(i, j)
    b_mag = np.linalg.norm(b_vec)

    # Helper function for linear Zeeman term
    def linear_zeeman_term(
        ops: list[sp.csr_matrix], scale_factor: complex
    ) -> sp.csr_matrix:
        x, y, z = ops
        return csr_scale(
            csr_add(
                csr_add(csr_scale(x, b_vec[0]), csr_scale(y, b_vec[1])),
                csr_scale(z, b_vec[2]),
            ),
            -complex(scale_factor / b_mag),
        )

    # Generate spin operators
    i_ops, j_ops = collective_ops_sparse(i, j)

    # Linear Zeeman terms
    hamiltonian_i = linear_zeeman_term(i_ops, a)
    hamiltonian_j = linear_zeeman_term(j_ops, b)

    # 2) spin-rotation
    i_dot_j = reduce(csr_add, map(sparse_matmul, i_ops, j_ops))
    hamiltonian_spin_rotation = csr_scale(i_dot_j, -c)

    # 3) quadrupole / spin-spin
    ij_sq = sparse_matmul(i_dot_j, i_dot_j)
    i_sq = reduce(csr_add, map(sparse_matmul, i_ops, i_ops))
    j_sq = reduce(csr_add, map(sparse_matmul, j_ops, j_ops))
    hamiltonian_quadrupole = csr_scale(
        csr_subtract(
            csr_add(csr_scale(ij_sq, 3), csr_scale(i_dot_j, 1.5)),
            sparse_matmul(i_sq, j_sq),
        ),
        (5 * d / ((2 * j - 1) * (2 * j + 3))),
    )
    return csr_add(
        csr_add(csr_add(hamiltonian_i, hamiltonian_j), hamiltonian_spin_rotation),
        hamiltonian_quadrupole,
    )
