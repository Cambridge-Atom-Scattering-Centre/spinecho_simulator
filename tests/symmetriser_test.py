from __future__ import annotations

import numpy as np
import pytest

from spinecho_sim.state import CoherentSpin, Spin
from spinecho_sim.util import csr_to_array, product_state, symmetrize


@pytest.mark.parametrize(
    ("input_spin", "expected_output"),
    [
        (
            Spin.from_iter(
                [
                    CoherentSpin(theta=0, phi=0),
                    CoherentSpin(theta=np.pi, phi=0),
                    CoherentSpin(theta=0, phi=0),
                ]
            ),
            np.array([0, 1, 1, 0, 1, 0, 0, 0] / np.sqrt(3), dtype=np.float64),
        ),
        (
            Spin.from_iter(
                [
                    CoherentSpin(theta=np.pi, phi=0),
                    CoherentSpin(theta=0, phi=0),
                    CoherentSpin(theta=0, phi=0),
                ]
            ),
            np.array([0, 1, 1, 0, 1, 0, 0, 0] / np.sqrt(3), dtype=np.float64),
        ),
        (
            Spin.from_iter(
                [
                    CoherentSpin(theta=np.pi, phi=0),
                    CoherentSpin(theta=0, phi=0),
                    CoherentSpin(theta=np.pi, phi=0),
                ]
            ),
            np.array([0, 0, 0, 1, 0, 1, 1, 0] / np.sqrt(3), dtype=np.float64),
        ),
    ],
)
def test_symmetrize(input_spin: Spin[tuple[int]], expected_output: np.ndarray) -> None:
    n = input_spin.n_stars
    p_sym = symmetrize(n)
    np.testing.assert_array_almost_equal(
        n + 1,
        np.linalg.matrix_rank(csr_to_array(p_sym)),
        err_msg="Symmetrize rank mismatch.",
    )

    output_state = csr_to_array(p_sym) @ product_state(input_spin._spins)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    output_state /= np.linalg.norm(output_state)  # normalize
    np.testing.assert_array_almost_equal(
        output_state,
        expected_output,
        err_msg="Symmetrization failed for input spin state.",
    )
