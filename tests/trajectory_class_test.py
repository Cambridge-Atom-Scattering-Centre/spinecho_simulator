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
    MonatomicTrajectoryList,
    TrajectoryList,
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
assert len(mtraj) == len([mstate1, mstate2])
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
assert len(dtraj) == len([dstate1, dstate2])
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


# --- TrajectoryList tests ---

# Create two MonatomicTrajectory objects
mtraj1 = MonatomicTrajectory(
    states=(mstate1, mstate2),
    state_type=MonatomicParticleState,
)
mtraj2 = MonatomicTrajectory(
    states=(mstate2, mstate1),
    state_type=MonatomicParticleState,
)

# Create a TrajectoryList from these
mlist = MonatomicTrajectoryList.from_trajectories([mtraj1, mtraj2])
assert isinstance(mlist, TrajectoryList)
assert len(mlist) == 2
assert mlist[0] == mtraj1
assert mlist[1] == mtraj2
assert tuple(mlist) == (mtraj1, mtraj2)

# Slicing returns a TrajectoryList of the same type
mlist_slice = mlist[:1]
assert isinstance(mlist_slice, TrajectoryList)
assert len(mlist_slice) == 1
assert mlist_slice[0] == mtraj1

# Iteration works
for traj in mlist:
    assert traj in {mtraj1, mtraj2}

# All trajectories must be of the same type
try:
    TrajectoryList.from_trajectories([mtraj1, dtraj])
except ValueError as e:
    assert "same type" in str(e)

# Empty list should raise ValueError
try:
    TrajectoryList.from_trajectories([])
except ValueError as e:
    assert "at least one trajectory" in str(e) or "No trajectories provided." in str(e)

print("All TrajectoryList tests passed.")
