from __future__ import annotations

import numpy as np

from spinecho_sim.state._displacement import ParticleDisplacement
from spinecho_sim.state._spin import CoherentSpin
from spinecho_sim.state._state import (
    DiatomicParticleState,
    MonatomicParticleState,
)
from spinecho_sim.state._trajectory import (
    BaseTrajectory,
    DiatomicTrajectory,
    MonatomicTrajectory,
)

# Create example spins and displacements
spin1 = CoherentSpin(theta=np.pi / 2, phi=0).as_generic()
spin2 = CoherentSpin(theta=np.pi / 3, phi=0).as_generic()
disp = ParticleDisplacement.from_cartesian(1.0, 1.0)

# Monatomic test states
mstate1 = MonatomicParticleState(
    spin_angular_momentum=spin1,
    displacement=disp,
    parallel_velocity=1.0,
    gyromagnetic_ratio=2.0,
)
mstate2 = MonatomicParticleState(
    spin_angular_momentum=spin2,
    displacement=disp,
    parallel_velocity=1.0,
    gyromagnetic_ratio=2.0,
)

# Diatomic test states
dstate1 = DiatomicParticleState(
    nuclear_angular_momentum=spin1,
    rotation_angular_momentum=spin2,
    displacement=disp,
    parallel_velocity=1.0,
)
dstate2 = DiatomicParticleState(
    nuclear_angular_momentum=spin2,
    rotation_angular_momentum=spin1,
    displacement=disp,
    parallel_velocity=1.0,
)

# MonatomicTrajectory test
mtraj = MonatomicTrajectory(
    states=(mstate1, mstate2),
    state_type=MonatomicParticleState,
)
assert len(mtraj) == 2
assert isinstance(mtraj[0], MonatomicParticleState)
assert mtraj[0].spin_angular_momentum == spin1
assert mtraj[1].spin_angular_momentum == spin2
assert isinstance(mtraj[:1], BaseTrajectory)
assert type(mtraj[:1]) is type(mtraj)
assert mtraj[:1].states == (mstate1,)

# DiatomicTrajectory test
dtraj = DiatomicTrajectory(
    states=(dstate1, dstate2),
    state_type=DiatomicParticleState,
)
assert len(dtraj) == 2
assert isinstance(dtraj[0], DiatomicParticleState)
assert dtraj[0].nuclear_angular_momentum == spin1
assert dtraj[1].nuclear_angular_momentum == spin2
assert isinstance(dtraj[:1], BaseTrajectory)
assert type(dtraj[:1]) is type(dtraj)
assert dtraj[:1].states == (dstate1,)

# from_states test
mtraj2 = MonatomicTrajectory.from_states(
    [mstate1, mstate2], state_type=MonatomicParticleState
)
assert isinstance(mtraj2, BaseTrajectory)
assert type(mtraj2) is type(mtraj)
assert mtraj2.states == (mstate1, mstate2)

print("All BaseTrajectory tests passed.")
