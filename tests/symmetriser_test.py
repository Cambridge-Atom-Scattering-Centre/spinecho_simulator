from __future__ import annotations

import numpy as np
import pytest

from spinecho_sim.util import symmetriser


@pytest.mark.parametrize(
    ("input_state", "expected_output"),
    [
        (
            np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64),
            np.array([0, 1, 1, 0, 1, 0, 0, 0] / np.sqrt(3), dtype=np.float64),
        ),
    ],
)
def test_symmetriser(input_state: np.ndarray, expected_output: np.ndarray) -> None:
    n = 3  # Number of qubits
    p_sym = symmetriser(n)
    np.testing.assert_array_almost_equal(n + 1, np.linalg.matrix_rank(p_sym.toarray()))
    output_state = p_sym @ input_state
    output_state /= np.linalg.norm(output_state)  # normalise
    np.testing.assert_array_almost_equal(output_state, expected_output)
