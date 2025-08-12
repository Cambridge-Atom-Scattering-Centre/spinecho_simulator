from __future__ import annotations

import numpy as np

from spinecho_sim.state import Spin
from spinecho_sim.state._spin import (
    get_bargmann_expectation_values,  # noqa: PLC2701
)

c1 = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
c2 = np.array([1, 0, 0], dtype=complex)
c3 = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)

print(" <Sx,Sy,Sz> =", get_bargmann_expectation_values(Spin.from_momentum_state(c1)))  # noqa: T201
print(" <Sx,Sy,Sz> =", get_bargmann_expectation_values(Spin.from_momentum_state(c2)))  # noqa: T201
print(" <Sx,Sy,Sz> =", get_bargmann_expectation_values(Spin.from_momentum_state(c3)))  # noqa: T201
