from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, overload, override

import numpy as np

from spinecho_sim.state._state import (
    BaseParticleState,
    DiatomicParticleState,
    MonatomicParticleState,
)

TState = TypeVar("TState", bound=BaseParticleState)


@dataclass(kw_only=True, frozen=True)
class BaseTrajectory[TState: BaseParticleState](ABC, Sequence[Any]):
    """A trajectory of a particle through the simulation."""

    states: tuple[TState, ...]  # All states in the single trajectory
    state_type: type[TState]  # Type of the states in the trajectory

    def __post_init__(self) -> None:
        if not self.states:
            msg = "Trajectory must contain at least one state."
            raise ValueError(msg)

        # all kinematics identical?
        initial_displacement = self.states[0].displacement
        initial_velocity = self.states[0].parallel_velocity

        if not all(state.displacement == initial_displacement for state in self.states):
            msg = "All states must share the same displacement."
            raise ValueError(msg)
        if not all(
            np.isclose(state.parallel_velocity, initial_velocity)
            for state in self.states
        ):
            msg = "All states must share the same parallel velocity."
            raise ValueError(msg)

        object.__setattr__(self, "displacement", initial_displacement)
        object.__setattr__(self, "parallel_velocity", float(initial_velocity))

    @staticmethod
    def from_states(
        states: Iterable[TState],
        state_type: type[TState],
    ) -> BaseTrajectory[TState]:
        """Create a BaseTrajectory from a list of ParticleStates."""
        states = tuple(states)
        assert states is not None, "No states provided."

        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )

        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return BaseTrajectory(
            states=states,
            state_type=state_type,
        )

    @override
    def __len__(self) -> int:
        return len(self.states)

    @overload
    def __getitem__(self: BaseTrajectory[TState], index: int) -> TState: ...

    @overload
    def __getitem__(self, index: slice | int) -> BaseTrajectory[TState]: ...

    @override
    def __getitem__(self, index: int | slice) -> TState | BaseTrajectory[TState]:
        if isinstance(index, int):
            state = self.states[index]
            # Use self.state_type to construct the correct type
            return self.state_type(
                **{
                    field: getattr(state, field)
                    for field in self.state_type.__dataclass_fields__
                }
            )
        return type(self)(
            states=self.states[index],
            state_type=self.state_type,
        )


MonatomicTrajectory = BaseTrajectory[MonatomicParticleState]
DiatomicTrajectory = BaseTrajectory[DiatomicParticleState]
GenericTrajectory = BaseTrajectory[BaseParticleState]

T = TypeVar(
    "T", bound=BaseTrajectory[Any]
)  # Use Any if you want to allow any BaseTrajectory
# Remove TState from the class generic parameters


@dataclass(kw_only=True, frozen=True)
class TrajectoryList[T: BaseTrajectory[Any]](Sequence[T]):
    """A list of trajectories."""

    trajectories: tuple[T, ...]

    def __post_init__(self) -> None:
        if not self.trajectories:
            msg = "TrajectoryList must contain at least one trajectory."
            raise ValueError(msg)

        # Check all trajectories have the same type
        first_type = type(self.trajectories[0])
        if not all(
            isinstance(trajectory, first_type) for trajectory in self.trajectories
        ):
            msg = "All trajectories must be of the same type."
            raise ValueError(msg)

    @staticmethod
    def from_trajectories(
        trajectories: Iterable[T],
    ) -> TrajectoryList[T]:
        """Create a TrajectoryList from a list of Trajectories."""
        trajectories = tuple(trajectories)
        assert trajectories is not None, "No trajectories provided."
        return TrajectoryList(trajectories=trajectories)

    @override
    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> TrajectoryList[T]: ...

    @override
    def __getitem__(self, index: int | slice) -> T | TrajectoryList[T]:
        if isinstance(index, slice):
            return TrajectoryList(trajectories=self.trajectories[index])
        return self.trajectories[index]

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self.trajectories)


MonatomicTrajectoryList = TrajectoryList[MonatomicTrajectory]
DiatomicTrajectoryList = TrajectoryList[DiatomicTrajectory]
GenericTrajectoryList = TrajectoryList[GenericTrajectory]
