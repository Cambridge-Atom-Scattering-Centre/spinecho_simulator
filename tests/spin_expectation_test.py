from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spinecho_sim.measurement import transverse_expectation


@pytest.mark.parametrize(
    ("c", "expected_jx", "expected_jy"),
    [
        (np.array([1, 0], dtype=np.complex128), 0.0, 0.0),  # |+z>
        (np.array([0, 1], dtype=np.complex128), 0.0, 0.0),  # |-z>
        (np.array([1, 1], dtype=np.complex128) / np.sqrt(2), 0.5, 0.0),  # |+x>
        (np.array([1, -1], dtype=np.complex128) / np.sqrt(2), -0.5, 0.0),  # |-x>
        (np.array([1, 1j], dtype=np.complex128) / np.sqrt(2), 0.0, 0.5),  # |+y>
        (np.array([1, -1j], dtype=np.complex128) / np.sqrt(2), 0.0, -0.5),  # |-y>
    ],
)
def test_spin_half_eigenstates(
    c: np.ndarray[Any, np.dtype[np.complex128]],
    expected_jx: float,
    expected_jy: float,
) -> None:
    hbar = 1.0
    jx, jy = transverse_expectation(c, hbar)
    np.testing.assert_array_almost_equal(
        jx,
        expected_jx,
        err_msg=f"Failed for state {c}, expected Jx={expected_jx}, got Jx={jx}",
    )
    np.testing.assert_array_almost_equal(
        jy,
        expected_jy,
        err_msg=f"Failed for state {c}, expected Jy={expected_jy}, got Jy={jy}",
    )


@pytest.mark.parametrize(
    ("c", "expected_jx", "expected_jy"),
    [
        (np.array([1, 0, 0], dtype=np.complex128), 0.0, 0.0),  # |z:+1>
        (np.array([0, 0, 1], dtype=np.complex128), 0.0, 0.0),  # |z:-1>
        (np.array([0, 1, 0], dtype=np.complex128), 0.0, 0.0),  # |z:0>
        (np.array([1, np.sqrt(2), 1], dtype=np.complex128) / 2, 1.0, 0.0),  # |x:+1>
        (np.array([-1, 0, 1], dtype=np.complex128) / np.sqrt(2), 0.0, 0.0),  # |x:0>
        (np.array([1, -np.sqrt(2), 1], dtype=np.complex128) / 2, -1.0, 0.0),  # |x:-1>
        (np.array([1, 1j * np.sqrt(2), -1], dtype=np.complex128) / 2, 0, 1),  # |y:+1>
        (np.array([1, 0, 1], dtype=np.complex128) / np.sqrt(2), 0, 0),  # |y:0>
        (np.array([1, -1j * np.sqrt(2), -1], dtype=np.complex128) / 2, 0, -1),  # |y:-1>
    ],
)
def test_spin_one_eigenstates(
    c: np.ndarray[Any, np.dtype[np.complex128]],
    expected_jx: float,
    expected_jy: float,
) -> None:
    hbar = 1.0
    jx, jy = transverse_expectation(c, hbar)
    np.testing.assert_array_almost_equal(
        jx,
        expected_jx,
        err_msg=f"Failed for state {c}, expected Jx={expected_jx}, got Jx={jx}",
    )
    np.testing.assert_array_almost_equal(
        jy,
        expected_jy,
        err_msg=f"Failed for state {c}, expected Jy={expected_jy}, got Jy={jy}",
    )
