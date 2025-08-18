from __future__ import annotations

from functools import cache, reduce

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
    validate_spin_quantum_number,
    verify_hermitian,
)


@cache
def single_spin_ops_sparse(
    s: float,
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Generate sparse matrices for single spin operators Sx, Sy, Sz."""
    dim = int(2 * s + 1)
    mj_values = np.arange(s, -s - 1, -1)
    j_z = csr_diags(mj_values, 0)

    # ladder matrices
    j_plus_lil = sp.lil_matrix((dim, dim), dtype=complex)
    for col in range(1, dim):  # col index corresponds to m = mvals[col]
        m = mj_values[col]  # m of the column state
        j_plus_lil[col - 1, col] = np.sqrt(s * (s + 1) - m * (m + 1))
    j_plus = sp.csr_matrix(j_plus_lil.tocsr())
    j_minus = csr_hermitian(j_plus)

    j_x = csr_scale(csr_add(j_plus, j_minus), 0.5)
    j_y = csr_scale(csr_subtract(j_plus, j_minus), -0.5j)
    assert verify_hermitian(j_x), "J_x is not Hermitian"
    assert verify_hermitian(j_y), "J_y is not Hermitian"
    assert verify_hermitian(j_z), "J_z is not Hermitian"
    return j_x, j_y, j_z


@cache
def collective_ops_sparse(
    i: float, j: float
) -> tuple[list[sp.csr_matrix], list[sp.csr_matrix]]:
    """Generate sparse matrices for single spin operators acting on the space of two spins."""
    i, j = round(i, ndigits=1), round(j, ndigits=1)
    validate_spin_quantum_number(i)
    validate_spin_quantum_number(j)
    i_x, i_y, i_z = single_spin_ops_sparse(i)
    j_x, j_y, j_z = single_spin_ops_sparse(j)

    dim_i, dim_j = int(2 * i + 1), int(2 * j + 1)
    identity_i = csr_eye(dim_i)
    identity_j = csr_eye(dim_j)

    operators_i = [csr_kron(op, identity_j) for op in (i_x, i_y, i_z)]
    operators_j = [csr_kron(identity_i, op) for op in (j_x, j_y, j_z)]
    assert all(verify_hermitian(op) for op in operators_i), (
        "Operators for spin i are not Hermitian"
    )
    assert all(verify_hermitian(op) for op in operators_j), (
        "Operators for spin j are not Hermitian"
    )
    return operators_i, operators_j


def zeeman_hamiltonian_dicke(
    ops: list[sp.csr_matrix], b_vec: np.ndarray
) -> sp.csr_matrix:
    """Generate and cache the Zeeman Hamiltonian for two spin systems."""
    x, y, z = ops
    return csr_add(
        csr_add(csr_scale(x, b_vec[0]), csr_scale(y, b_vec[1])),
        csr_scale(z, b_vec[2]),
    )


@cache
def spin_rotation_hamiltonian_dicke(i: float, j: float) -> sp.csr_matrix:
    """Generate the spin-rotation Hamiltonian for two spin systems."""
    i, j = round(i, ndigits=1), round(j, ndigits=1)
    i_ops, j_ops = collective_ops_sparse(i, j)
    return reduce(csr_add, map(sparse_matmul, i_ops, j_ops))


@cache
def quadrupole_hamiltonian_dicke(i: float, j: float) -> sp.csr_matrix:
    """Generate the quadrupole Hamiltonian for two spin systems."""
    i, j = round(i, ndigits=1), round(j, ndigits=1)
    i_ops, j_ops = collective_ops_sparse(i, j)
    i_dot_j = spin_rotation_hamiltonian_dicke(i, j)
    ij_sq = sparse_matmul(i_dot_j, i_dot_j)
    i_sq = reduce(csr_add, map(sparse_matmul, i_ops, i_ops))
    j_sq = reduce(csr_add, map(sparse_matmul, j_ops, j_ops))
    return csr_subtract(
        csr_add(csr_scale(ij_sq, 3), csr_scale(i_dot_j, 1.5)),
        sparse_matmul(i_sq, j_sq),
    )


@cache
def cache_terms_hamiltonian_dicke(
    i: float, j: float, c: float, d: float
) -> sp.csr_matrix:
    """Generate the cache terms Hamiltonian for two spin systems."""
    i, j = round(i, ndigits=1), round(j, ndigits=1)
    # 2) spin-rotation
    hamiltonian_spin_rotation = csr_scale(spin_rotation_hamiltonian_dicke(i, j), c)
    assert verify_hermitian(hamiltonian_spin_rotation), (
        "Spin-rotation Hamiltonian is not Hermitian"
    )

    # 3) quadrupole / spin-spin
    denominator = (2 * j - 1) * (2 * j + 3)
    assert denominator != 0, f"Invalid denominator for quadrupole scaling: j={j}."
    hamiltonian_quadrupole = csr_scale(
        quadrupole_hamiltonian_dicke(i, j),
        (5 * d / denominator),
    )
    assert verify_hermitian(hamiltonian_quadrupole), (
        "Quadrupole Hamiltonian is not Hermitian"
    )
    return csr_subtract(hamiltonian_quadrupole, hamiltonian_spin_rotation)


def diatomic_hamiltonian_dicke(
    i: float,
    j: float,
    coefficients: tuple[float, float, float, float],
    b_vec: np.ndarray,
) -> sp.csr_matrix:
    """Generate the Ramsey Hamiltonian as a sparse matrix."""
    a, b, c, d = coefficients
    # Generate spin operators
    i, j = round(i, ndigits=1), round(j, ndigits=1)
    i_ops, j_ops = collective_ops_sparse(i, j)

    # Linear Zeeman terms
    hamiltonian_i = csr_scale(zeeman_hamiltonian_dicke(i_ops, b_vec), -a)
    hamiltonian_j = csr_scale(zeeman_hamiltonian_dicke(j_ops, b_vec), -b)

    return csr_add(
        csr_add(hamiltonian_i, hamiltonian_j), cache_terms_hamiltonian_dicke(i, j, c, d)
    )
