from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload, override

import numpy as np

from spinecho_sim.state import (
    MonatomicParticleState,
    ParticleDisplacementList,
    ParticleState,
    Spin,
)
from spinecho_sim.state._spin import EmptySpinList, EmptySpinListList

if TYPE_CHECKING:
    from spinecho_sim.state import (
        GenericSpinList,
        ParticleDisplacement,
    )


@dataclass(frozen=True, kw_only=True)
class Trajectory(Sequence[Any]):
    """A trajectory of a diatomic particle through the simulation."""

    _spin_angular_momentum: GenericSpinList
    _rotational_angular_momentum: GenericSpinList

    displacement: ParticleDisplacement
    parallel_velocity: float

    @staticmethod
    def from_states(
        states: Iterable[ParticleState],
    ) -> Trajectory:
        """Create a Trajectory from a list of ParticleStates."""
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return Trajectory(
            _spin_angular_momentum=Spin.from_iter(s.spin for s in states),
            _rotational_angular_momentum=Spin.from_iter(
                s.rotational_angular_momentum for s in states
            ),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @override
    def __len__(self) -> int:
        return self.spin.shape[0]

    @property
    def spin(self) -> GenericSpinList:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> GenericSpinList:
        return self._rotational_angular_momentum

    @overload
    def __getitem__(self: Trajectory, index: int) -> ParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> Trajectory: ...

    @override
    def __getitem__(self, index: int | slice) -> ParticleState | Trajectory:
        if isinstance(index, int):
            return ParticleState(
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return Trajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectory(Trajectory):
    _rotational_angular_momentum: EmptySpinList = field(
        init=False
    )  # Automatically set later

    def __post_init__(self) -> None:
        """Automatically set rotational angular momentum to an EmptySpinList with the same shape as spin."""
        object.__setattr__(
            self,
            "_rotational_angular_momentum",
            EmptySpinList(self._spin_angular_momentum.shape),
        )

    @staticmethod
    @override
    def from_states(
        states: Iterable[MonatomicParticleState] | Iterable[ParticleState],
    ) -> MonatomicTrajectory:
        """Create a Trajectory from a list of ParticleStates."""
        mono_states = [s for s in states if isinstance(s, MonatomicParticleState)]
        assert mono_states is not None, "No MonatomicParticleState instances provided."

        velocities = np.array([state.parallel_velocity for state in mono_states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in mono_states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return MonatomicTrajectory(
            _spin_angular_momentum=Spin.from_iter(s.spin for s in mono_states),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpinList:
        return EmptySpinList(self.rotational_angular_momentum.shape)

    @overload
    def __getitem__(
        self: MonatomicTrajectory, index: int
    ) -> MonatomicParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> MonatomicTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> MonatomicParticleState | MonatomicTrajectory:
        if isinstance(index, int):
            return MonatomicParticleState(
                _spin_angular_momentum=self.spin[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class TrajectoryList(Sequence[Trajectory]):
    """A list of diatomic trajectories."""

    _spin_angular_momentum: Spin[tuple[int, int, int]]
    _rotational_angular_momentum: Spin[tuple[int, int, int]]
    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self._spin_angular_momentum.shape[0]
            or self.parallel_velocities.size
            != self.rotational_angular_momentum.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)

    @property
    def spin(self) -> Spin[tuple[int, int, int]]:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return self._rotational_angular_momentum

    @staticmethod
    def from_trajectories(
        trajectories: Iterable[Trajectory],
    ) -> TrajectoryList:
        """Create a DiatomicTrajectoryList from a list of DiatomicTrajectories."""
        nuclear_spins = Spin.from_iter(t.spin for t in trajectories)
        rotational_spins = Spin.from_iter(
            t.rotational_angular_momentum for t in trajectories
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return TrajectoryList(
            _spin_angular_momentum=nuclear_spins,
            _rotational_angular_momentum=rotational_spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @override
    def __len__(self) -> int:
        return len(self.parallel_velocities)

    @overload
    def __getitem__(self, index: int) -> Trajectory: ...
    @overload
    def __getitem__(self, index: slice) -> TrajectoryList: ...

    @override
    def __getitem__(self, index: int | slice) -> Trajectory | TrajectoryList:
        if isinstance(index, slice):
            return TrajectoryList(
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return Trajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[Trajectory]:
        for i in range(len(self)):
            yield Trajectory(
                _spin_angular_momentum=self.spin[i],
                _rotational_angular_momentum=self.rotational_angular_momentum[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectoryList(TrajectoryList):
    """A list of monatomic trajectories."""

    _rotational_angular_momentum: EmptySpinListList = field(
        init=False
    )  # Automatically set later

    @override
    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spin.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)
        """Automatically set rotational angular momentum to an EmptySpinList with the same shape as spin."""
        object.__setattr__(
            self,
            "_rotational_angular_momentum",
            EmptySpinListList(self._spin_angular_momentum.shape),
        )

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]: ...

    @staticmethod
    @override
    def from_trajectories(
        trajectories: Iterable[Trajectory],
    ) -> MonatomicTrajectoryList:
        """Create a MonatomicTrajectoryList from a list of MonatomicTrajectories."""
        mono_trajectories = [
            trajectory
            for trajectory in trajectories
            if isinstance(trajectory, MonatomicTrajectory)
        ]
        assert mono_trajectories, "No MonatomicTrajectory instances provided."

        spins = Spin.from_iter(t.spin for t in mono_trajectories)
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in mono_trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in mono_trajectories])
        return MonatomicTrajectoryList(
            _spin_angular_momentum=spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @overload
    def __getitem__(self, index: int) -> MonatomicTrajectory: ...
    @overload
    def __getitem__(self, index: slice) -> MonatomicTrajectoryList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> MonatomicTrajectory | MonatomicTrajectoryList:
        if isinstance(index, slice):
            return MonatomicTrajectoryList(
                _spin_angular_momentum=self.spin[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[MonatomicTrajectory]:
        for i in range(len(self)):
            yield MonatomicTrajectory(
                _spin_angular_momentum=self.spin[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )
