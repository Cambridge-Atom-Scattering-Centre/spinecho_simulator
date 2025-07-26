from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def transverse_expectation_old(
    state_coefficients: NDArray[np.complex128], hbar: float = 1.0
) -> tuple[float, float, float]:
    """Return the expectation values of S_x, S_y, and S_z for a given state vector."""
    # order of state components is m_s=[j, j-1, ..., -j+1, -j]
    two_j = state_coefficients.size - 1
    j = two_j / 2

    m = np.arange(-j, j)  # length 2j   (stops at j-1)
    factors = np.sqrt((j - m) * (j + m + 1))  # length 2j
    inner = np.conjugate(state_coefficients[:-1]) * state_coefficients[1:] * factors

    j_plus = hbar * inner.sum()
    jx = float(j_plus.real)
    jy = float(j_plus.imag)

    # S_z operator
    m_z = np.arange(j, -j - 1, -1)
    print(m_z)
    jz = float(hbar * np.sum(np.abs(state_coefficients) ** 2 * m_z))

    return jx, jy, jz


@cache
def _j_plus_factors(
    two_j: int, hbar: float = 1.0
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Return a sparse array of J_+ ladder factors."""
    j = two_j / 2
    m = np.arange(-j, j)  # length 2j   (stops at j-1)
    return hbar * np.sqrt((j - m) * (j + m + 1))


def transverse_expectation(
    state_coefficients: NDArray[np.complex128], hbar: float = 1.0
) -> tuple[float, float, float]:
    """Return the expectation values of S_x, S_y, and S_z for a given state vector using cached arrays."""
    two_j = state_coefficients.size - 1
    factors = _j_plus_factors(two_j, hbar)  # sparse array

    inner = np.conjugate(state_coefficients[:-1]) * state_coefficients[1:] * factors
    j_plus = inner.sum()

    jx = float(j_plus.real)
    jy = float(j_plus.imag)

    m_z = np.arange(two_j / 2, -two_j / 2 - 1, -1, dtype=np.float64)
    jz = float(hbar * np.sum(np.abs(state_coefficients) ** 2 * m_z))
    return jx, jy, jz
