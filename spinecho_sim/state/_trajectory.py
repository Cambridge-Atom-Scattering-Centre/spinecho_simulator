from __future__ import annotations

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
from spinecho_sim.state._spin import EmptySpin, EmptySpinList, EmptySpinListList

if TYPE_CHECKING:
    from spinecho_sim.state import (
        GenericSpinList,
        ParticleDisplacement,
    )


@dataclass(frozen=True, kw_only=True)
class DiatomicTrajectory(Sequence[Any]):
    """A trajectory of a diatomic particle through the simulation."""

    _spin_angular_momentum: GenericSpinList
    _rotational_angular_momentum: GenericSpinList

    displacement: ParticleDisplacement
    parallel_velocity: float

    @staticmethod
    def from_states(
        states: Iterable[DiatomicParticleState],
    ) -> DiatomicTrajectory:
        """Create a Trajectory from a list of ParticleStates."""
        velocities = np.array([state.parallel_velocity for state in states])
        assert np.allclose(velocities, velocities[0]), (
            "All states must have the same velocity."
        )
        displacements = [state.displacement for state in states]
        assert all(d == displacements[0] for d in displacements), (
            "All states must have the same displacement."
        )

        return DiatomicTrajectory(
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
    def __getitem__(self: DiatomicTrajectory, index: int) -> DiatomicParticleState: ...

    @overload
    def __getitem__(self, index: slice | int) -> DiatomicTrajectory: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> DiatomicParticleState | DiatomicTrajectory:
        if isinstance(index, int):
            return DiatomicParticleState(
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return DiatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectory(DiatomicTrajectory):
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
            _spin_angular_momentum=Spin.from_iter(s.spin for s in mono_states),
            _rotational_angular_momentum=EmptySpinList(
                Spin.from_iter(s.spin for s in mono_states).shape
            ),
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
                _rotational_angular_momentum=EmptySpin(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )

        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=EmptySpinList(self.spin[index].shape),
            displacement=self.displacement,
            parallel_velocity=self.parallel_velocity,
        )


@dataclass(kw_only=True, frozen=True)
class DiatomicTrajectoryList(Sequence[DiatomicTrajectory]):
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
        trajectories: Iterable[DiatomicTrajectory],
    ) -> DiatomicTrajectoryList:
        """Create a DiatomicTrajectoryList from a list of DiatomicTrajectories."""
        nuclear_spins = Spin.from_iter(t.spin for t in trajectories)
        rotational_spins = Spin.from_iter(
            t.rotational_angular_momentum for t in trajectories
        )
        displacements = ParticleDisplacementList.from_displacements(
            t.displacement for t in trajectories
        )
        parallel_velocities = np.array([t.parallel_velocity for t in trajectories])
        return DiatomicTrajectoryList(
            _spin_angular_momentum=nuclear_spins,
            _rotational_angular_momentum=rotational_spins,
            displacements=displacements,
            parallel_velocities=parallel_velocities,
        )

    @override
    def __len__(self) -> int:
        return len(self.parallel_velocities)

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
                _spin_angular_momentum=self.spin[index],
                _rotational_angular_momentum=self.rotational_angular_momentum[index],
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return DiatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=self.rotational_angular_momentum[index],
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[DiatomicTrajectory]:
        for i in range(len(self)):
            yield DiatomicTrajectory(
                _spin_angular_momentum=self.spin[i],
                _rotational_angular_momentum=self.rotational_angular_momentum[i],
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )


@dataclass(frozen=True, kw_only=True)
class MonatomicTrajectoryList(DiatomicTrajectoryList):
    """A list of monatomic trajectories."""

    @override
    def __post_init__(self) -> None:
        if (
            self.parallel_velocities.ndim != 1
            or self.parallel_velocities.shape != self.displacements.shape
            or self.parallel_velocities.size != self.spin.shape[0]
        ):
            msg = "Spins must be a 2D array, parallel velocities and displacements must be 1D arrays, and their shapes must match."
            raise ValueError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]: ...

    @staticmethod
    @override
    def from_trajectories(
        trajectories: Iterable[DiatomicTrajectory],
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
            _rotational_angular_momentum=EmptySpinListList(spins.shape),
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
                _rotational_angular_momentum=EmptySpinListList(self.spin[index].shape),
                displacements=self.displacements[index],
                parallel_velocities=self.parallel_velocities[index],
            )
        return MonatomicTrajectory(
            _spin_angular_momentum=self.spin[index],
            _rotational_angular_momentum=EmptySpinList(self.spin[index].shape),
            displacement=self.displacements[index],
            parallel_velocity=self.parallel_velocities[index],
        )

    @override
    def __iter__(self) -> Iterator[MonatomicTrajectory]:
        for i in range(len(self)):
            yield MonatomicTrajectory(
                _spin_angular_momentum=self.spin[i],
                _rotational_angular_momentum=EmptySpinList(self.spin[i].shape),
                displacement=self.displacements[i],
                parallel_velocity=self.parallel_velocities[i],
            )
