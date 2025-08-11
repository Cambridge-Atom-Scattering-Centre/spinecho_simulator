from __future__ import annotations

import numpy as np
import pytest

from spinecho_sim.state import (
    CoherentSpin,
    DiatomicParticleState,
    DiatomicTrajectory,
    MonatomicParticleState,
    MonatomicTrajectory,
    MonatomicTrajectoryList,
    ParticleDisplacement,
    Trajectory,
    TrajectoryList,
)

# Create example spins and displacements
spin_1 = CoherentSpin(theta=np.pi / 2, phi=0).as_generic()
spin_2 = CoherentSpin(theta=np.pi / 3, phi=0).as_generic()
disp = ParticleDisplacement.from_cartesian(1.0, 1.0)

# Monatomic test states
mstate1 = MonatomicParticleState(
    spin_angular_momentum=spin_1,
    displacement=disp,
    parallel_velocity=1.0,
    gyromagnetic_ratio=2.0,
)
mstate2 = MonatomicParticleState(
    spin_angular_momentum=spin_2,
    displacement=disp,
    parallel_velocity=1.0,
    gyromagnetic_ratio=2.0,
)

# Diatomic test states
diatomic_state_1 = DiatomicParticleState(
    nuclear_angular_momentum=spin_1,
    rotation_angular_momentum=spin_2,
    displacement=disp,
    parallel_velocity=1.0,
)
diatomic_state_2 = DiatomicParticleState(
    nuclear_angular_momentum=spin_2,
    rotation_angular_momentum=spin_1,
    displacement=disp,
    parallel_velocity=1.0,
)

# MonatomicTrajectory test
monatomic_trajectory = MonatomicTrajectory(
    states=(mstate1, mstate2),
    state_type=MonatomicParticleState,
)
assert len(monatomic_trajectory) == len([mstate1, mstate2])
assert isinstance(monatomic_trajectory[0], MonatomicParticleState)
assert monatomic_trajectory[0].spin_angular_momentum == spin_1
assert monatomic_trajectory[1].spin_angular_momentum == spin_2
assert isinstance(monatomic_trajectory[:1], Trajectory)
assert type(monatomic_trajectory[:1]) is type(monatomic_trajectory)
assert monatomic_trajectory[:1].states == (mstate1,)

# DiatomicTrajectory test
diatomic_trajectory = DiatomicTrajectory(
    states=(diatomic_state_1, diatomic_state_2),
    state_type=DiatomicParticleState,
)
assert len(diatomic_trajectory) == len([diatomic_state_1, diatomic_state_2])
assert isinstance(diatomic_trajectory[0], DiatomicParticleState)
assert diatomic_trajectory[0].nuclear_angular_momentum == spin_1
assert diatomic_trajectory[1].nuclear_angular_momentum == spin_2
assert isinstance(diatomic_trajectory[:1], Trajectory)
assert type(diatomic_trajectory[:1]) is type(diatomic_trajectory)
assert diatomic_trajectory[:1].states == (diatomic_state_1,)

# from_states test
monatomic_trajectory_2 = MonatomicTrajectory.from_states(
    [mstate1, mstate2], state_type=MonatomicParticleState
)
assert isinstance(monatomic_trajectory_2, Trajectory)
assert type(monatomic_trajectory_2) is type(monatomic_trajectory)
assert monatomic_trajectory_2.states == (mstate1, mstate2)

print("All BaseTrajectory tests passed.")


# --- TrajectoryList tests ---

# Create two MonatomicTrajectory objects
monatomic_trajectory_1 = MonatomicTrajectory(
    states=(mstate1, mstate2),
    state_type=MonatomicParticleState,
)
monatomic_trajectory_2 = MonatomicTrajectory(
    states=(mstate2, mstate1),
    state_type=MonatomicParticleState,
)

# Create a TrajectoryList from these
monatomic_trajectory_list = MonatomicTrajectoryList.from_trajectories(
    [monatomic_trajectory_1, monatomic_trajectory_2]
)
assert isinstance(monatomic_trajectory_list, TrajectoryList)
assert len(monatomic_trajectory_list) == 2
assert monatomic_trajectory_list[0] == monatomic_trajectory_1
assert monatomic_trajectory_list[1] == monatomic_trajectory_2
assert tuple(monatomic_trajectory_list) == (
    monatomic_trajectory_1,
    monatomic_trajectory_2,
)

# Slicing returns a TrajectoryList of the same type
monatomic_trajectory_list_slice = monatomic_trajectory_list[:1]
assert isinstance(monatomic_trajectory_list_slice, TrajectoryList)
assert len(monatomic_trajectory_list_slice) == 1
assert monatomic_trajectory_list_slice[0] == monatomic_trajectory_1

# Iteration works
for trajectory in monatomic_trajectory_list:
    assert trajectory in {monatomic_trajectory_1, monatomic_trajectory_2}

# All trajectories must be of the same type
with pytest.raises(ValueError, match="state type"):
    TrajectoryList.from_trajectories([monatomic_trajectory_1, diatomic_trajectory])

# Empty list should raise ValueError
with pytest.raises(
    ValueError, match=r"at least one trajectory|No trajectories provided."
):
    TrajectoryList.from_trajectories([])

print("All TrajectoryList tests passed.")
