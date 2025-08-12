from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload, override

import numpy as np

from spinecho_sim.state import (
    DiatomicParticleState,
    MonatomicParticleState,
    ParticleDisplacementList,
    Spin,
)

if TYPE_CHECKING:
    from spinecho_sim.state import (
        GenericSpinList,
        ParticleDisplacement,
    )


@dataclass(kw_only=True, frozen=True)
class Trajectory(ABC, Sequence[Any]):
    """A trajectory of a particle through the simulation."""

    displacement: ParticleDisplacement
    parallel_velocity: float

    @staticmethod
    @abstractmethod
    def from_states(
        states: Iterable[MonatomicParticleState] | Iterable[DiatomicParticleState],
    ) -> Trajectory: ...

    @override
    def __len__(self) -> int:
        return self.spins[0].shape[0]

    @property
    @abstractmethod
    def spins(self) -> tuple[GenericSpinList, ...]: ...


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectory(Trajectory):
    spin_angular_momentum: GenericSpinList

    @staticmethod
    @override
    def from_states(
        states: Iterable[MonatomicParticleState] | Iterable[DiatomicParticleState],
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
            spin_angular_momentum=Spin.from_iter(
                spin for s in mono_states for spin in s.spins
            ),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @property
    @override
    def spins(self) -> tuple[GenericSpinList, ...]:
        return (self.spin_angular_momentum,)

    @overload
    def __getitem__(self: Trajectory, index: int) -> MonatomicParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> MonatomicTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> MonatomicParticleState | MonatomicTrajectory:
        if isinstance(index, int):
            return MonatomicParticleState(
                spin_angular_momentum=self.spins[0][index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return MonatomicTrajectory(
            spin_angular_momentum=self.spins[0][index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(frozen=True, kw_only=True)
class DiatomicTrajectory(Trajectory):
    nuclear_angular_momentum: GenericSpinList
    rotational_angular_momentum: GenericSpinList

    @staticmethod
    @override
    def from_states(
        states: Iterable[MonatomicParticleState] | Iterable[DiatomicParticleState],
    ) -> DiatomicTrajectory:
        """Create a Trajectory from a list of ParticleStates."""
        dia_states = [s for s in states if isinstance(s, DiatomicParticleState)]
        assert dia_states is not None, "No DiatomicParticleState instances provided."
        velocities = np.array([state.parallel_velocity for state in dia_states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in dia_states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return DiatomicTrajectory(
            nuclear_angular_momentum=Spin.from_iter(
                s.nuclear_angular_momentum for s in dia_states
            ),
            rotational_angular_momentum=Spin.from_iter(
                s.rotational_angular_momentum for s in dia_states
            ),
            displacement=displacements[0],
            parallel_velocity=velocities[0],
        )

    @property
    @override
    def spins(self) -> tuple[GenericSpinList, ...]:
        return (self.nuclear_angular_momentum, self.rotational_angular_momentum)

    @overload
    def __getitem__(self: Trajectory, index: int) -> DiatomicParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> DiatomicTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> DiatomicParticleState | DiatomicTrajectory:
        if isinstance(index, int):
            return DiatomicParticleState(
                nuclear_angular_momentum=self.spins[0][index],
                rotational_angular_momentum=self.spins[1][index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return DiatomicTrajectory(
            nuclear_angular_momentum=self.spins[0][index],
            rotational_angular_momentum=self.spins[1][index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class TrajectoryList(ABC, Sequence[Trajectory]):
    """A list of trajectories."""

    displacements: ParticleDisplacementList
    parallel_velocities: np.ndarray[Any, np.dtype[np.floating]]

    @abstractmethod
    def __post_init__(self) -> None: ...

    @property
    @abstractmethod
    def spins(self) -> tuple[Spin[tuple[int, int, int]], ...]: ...

    @staticmethod
    @abstractmethod
    def from_trajectories(
        trajectories: Iterable[MonatomicTrajectory] | Iterable[DiatomicTrajectory],
    ) -> TrajectoryList: ...

    @override
    def __len__(self) -> int:
        return len(self.parallel_velocities)

    @override
    @abstractmethod
    def __iter__(self) -> Iterator[Trajectory]: ...


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectoryList(TrajectoryList):
    """A list of monatomic trajectories."""

    spin_angular_momentum: Spin[tuple[int, int, int]]

    @override
    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spin_angular_momentum.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)

    @property
    @override
    def spins(self) -> tuple[Spin[tuple[int, int, int]], ...]:
        return (self.spin_angular_momentum,)

    @staticmethod
    @override
    def from_trajectories(
        trajectories: Iterable[MonatomicTrajectory] | Iterable[DiatomicTrajectory],
    ) -> MonatomicTrajectoryList:
        """Create a MonatomicTrajectoryList from a list of MonatomicTrajectories."""
        mono_trajectories = [
            trajectory
            for trajectory in trajectories
            if isinstance(trajectory, MonatomicTrajectory)
        ]
        assert mono_trajectories, "No MonatomicTrajectory instances provided."

        spins = Spin.from_iter(t.spin_angular_momentum for t in mono_trajectories)
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in mono_trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in mono_trajectories])
        return MonatomicTrajectoryList(
            spin_angular_momentum=spins,
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
                spin_angular_momentum=self.spin_angular_momentum[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return MonatomicTrajectory(
            spin_angular_momentum=self.spin_angular_momentum[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[MonatomicTrajectory]:
        for i in range(len(self)):
            yield MonatomicTrajectory(
                spin_angular_momentum=self.spin_angular_momentum[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


@dataclass(frozen=True, kw_only=True)
class DiatomicTrajectoryList(TrajectoryList):
    """A list of diatomic trajectories."""

    nuclear_angular_momentum: Spin[tuple[int, int, int]]
    rotational_angular_momentum: Spin[tuple[int, int, int]]

    @override
    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.nuclear_angular_momentum.shape[0]
            or self.parallel_velocities.size
            != self.rotational_angular_momentum.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)

    @property
    @override
    def spins(self) -> tuple[Spin[tuple[int, int, int]], ...]:
        return (self.nuclear_angular_momentum, self.rotational_angular_momentum)

    @staticmethod
    @override
    def from_trajectories(
        trajectories: Iterable[MonatomicTrajectory] | Iterable[DiatomicTrajectory],
    ) -> DiatomicTrajectoryList:
        """Create a DiatomicTrajectoryList from a list of DiatomicTrajectories."""
        dia_trajectories = [
            trajectory
            for trajectory in trajectories
            if isinstance(trajectory, DiatomicTrajectory)
        ]
        assert dia_trajectories, "No DiatomicTrajectory instances provided."

        nuclear_spins = Spin.from_iter(
            t.nuclear_angular_momentum for t in dia_trajectories
        )
        rotational_spins = Spin.from_iter(
            t.rotational_angular_momentum for t in dia_trajectories
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in dia_trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in dia_trajectories])
        return DiatomicTrajectoryList(
            nuclear_angular_momentum=nuclear_spins,
            rotational_angular_momentum=rotational_spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @overload
    def __getitem__(self, index: int) -> DiatomicTrajectory: ...
    @overload
    def __getitem__(self, index: slice) -> DiatomicTrajectoryList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> DiatomicTrajectory | DiatomicTrajectoryList:
        if isinstance(index, slice):
            return DiatomicTrajectoryList(
                nuclear_angular_momentum=self.nuclear_angular_momentum[index],
                rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return DiatomicTrajectory(
            nuclear_angular_momentum=self.nuclear_angular_momentum[index],
            rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[DiatomicTrajectory]:
        for i in range(len(self)):
            yield DiatomicTrajectory(
                nuclear_angular_momentum=self.nuclear_angular_momentum[i],
                rotational_angular_momentum=self.rotational_angular_momentum[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )
