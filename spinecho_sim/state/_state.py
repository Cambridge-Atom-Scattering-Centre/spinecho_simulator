from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, override

from spinecho_sim.state._displacement import ParticleDisplacement
from spinecho_sim.state._spin import EmptySpin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spinecho_sim.state._spin import GenericSpin


@dataclass(frozen=True, kw_only=True)
class ParticleState:
    """State of a diatomic particle."""

    displacement: ParticleDisplacement = field(default_factory=ParticleDisplacement)
    parallel_velocity: float
    _spin_angular_momentum: GenericSpin
    _rotational_angular_momentum: GenericSpin

    @property
    def spin(self) -> GenericSpin:
        return self._spin_angular_momentum

    @property
    def rotational_angular_momentum(self) -> GenericSpin:
        return self._rotational_angular_momentum

    @abstractmethod
    def as_coherent(self) -> Sequence[CoherentParticleState]:
        return [
            CoherentParticleState(
                _spin_angular_momentum=spin.as_generic(),
                _rotational_angular_momentum=rot.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )
            for spin, rot in product(
                self._spin_angular_momentum.flat_iter(),
                self._rotational_angular_momentum.flat_iter(),
            )
        ]


@dataclass(frozen=True, kw_only=True)
class CoherentParticleState(ParticleState):
    def __post_init__(self) -> None:
        assert self.spin.size == 1
        assert self.rotational_angular_momentum.size == 1


@dataclass(frozen=True, kw_only=True)
class MonatomicParticleState(ParticleState):
    gyromagnetic_ratio: float = -2.04e8  # default value for 3He
    _rotational_angular_momentum: EmptySpin = field(init=False)

    def __post_init__(self) -> None:
        """Automatically set rotational angular momentum to an EmptySpin with the same shape as spin."""
        object.__setattr__(self, "_rotational_angular_momentum", EmptySpin())

    @property
    @override
    def spin(self) -> GenericSpin:
        return self._spin_angular_momentum

    @override
    def as_coherent(self) -> Sequence[CoherentMonatomicParticleState]:
        return [
            CoherentMonatomicParticleState(
                _spin_angular_momentum=s.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
                gyromagnetic_ratio=self.gyromagnetic_ratio,
            )
            for s in self._spin_angular_momentum.flat_iter()
        ]

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpin:
        return EmptySpin()


@dataclass(frozen=True, kw_only=True)
class CoherentMonatomicParticleState(MonatomicParticleState, CoherentParticleState):
    _rotational_angular_momentum: EmptySpin = field(init=False)

    def __post_init__(self) -> None:
        assert self._spin_angular_momentum.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )
        """Automatically set rotational angular momentum to an EmptySpin with the same shape as spin."""
        object.__setattr__(self, "_rotational_angular_momentum", EmptySpin())

    @property
    @override
    def rotational_angular_momentum(self) -> GenericSpin:
        """Ensure rotational_angular_momentum uses MonatomicParticleState's implementation."""
        return super(MonatomicParticleState, self).rotational_angular_momentum
