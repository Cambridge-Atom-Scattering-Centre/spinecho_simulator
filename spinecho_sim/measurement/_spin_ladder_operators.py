from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def transverse_expectation(
    c: NDArray[np.complex128], hbar: float = 1.0
) -> tuple[float, float]:
    c = np.asarray(c, dtype=np.complex128)
    two_j = c.size - 1
    j = two_j / 2

    m = np.arange(-j, j)  # length 2j   (stops at j-1)
    factors = np.sqrt((j - m) * (j + m + 1))  # length 2j
    inner = np.conjugate(c[:-1]) * c[1:] * factors

    j_plus = hbar * inner.sum()
    jx = j_plus.real
    jy = j_plus.imag
    return jx, jy
