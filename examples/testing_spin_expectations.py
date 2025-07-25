from __future__ import annotations

import numpy as np

from spinecho_sim.state import CoherentSpin, Spin
from spinecho_sim.state._spin import get_spin_expectation_values

if __name__ == "__main__":
    test = Spin.from_iter([CoherentSpin(theta=np.pi, phi=0)])
    print(test.momentum_states)
    print(get_spin_expectation_values(test.momentum_states))
    print(test.expectation_values)
