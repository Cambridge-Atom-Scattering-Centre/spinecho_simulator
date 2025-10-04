from __future__ import annotations

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]

from spinecho_sim.molecule.hamiltonian_dicke import (
    build_collective_operators,
    build_zeeman_hamiltonian_dicke,
)
from spinecho_sim.util import csr_scale


def test_csr_scale() -> None:
    mat = sp.csr_matrix([[1, 2], [3, 4]])
    scaled = csr_scale(mat, 2)
    expected = sp.csr_matrix([[2, 4], [6, 8]])
    assert (scaled != expected).nnz == 0  # pyright: ignore[reportAttributeAccessIssue]


def test_csr_scale_zero() -> None:
    i, j = 1, 1
    i_ops, _j_ops = build_collective_operators(two_i=2 * i, two_j=2 * j)
    b_vec = np.array([0.2, 0.2, 1])
    mat = build_zeeman_hamiltonian_dicke(operator_list=i_ops, b_vec=b_vec)
    scaled = csr_scale(mat, complex(0))
    expected = sp.csr_matrix(mat.shape, dtype=mat.dtype)  # pyright: ignore[reportUnknownArgumentType]
    assert (scaled != expected).nnz == 0  # pyright: ignore[reportAttributeAccessIssue]
