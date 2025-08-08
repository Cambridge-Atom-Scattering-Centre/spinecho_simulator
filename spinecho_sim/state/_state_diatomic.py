from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spinecho_sim.state._displacement import ParticleDisplacement

if TYPE_CHECKING:  # avoid circular imports at run-time
    from spinecho_sim.state._spin import GenericSpin


@dataclass(frozen=True, kw_only=True)
class DiatomicParticleState:
    nuclear_angular_momentum: GenericSpin  # nuclear spin
    rotation_angular_momentum: GenericSpin  # rotational angular momentum
    displacement: ParticleDisplacement = field(default_factory=ParticleDisplacement)
    parallel_velocity: float

    @property
    def i(self) -> GenericSpin:
        return self.nuclear_angular_momentum

    @property
    def j(self) -> GenericSpin:
        return self.rotation_angular_momentum


@dataclass(frozen=True, kw_only=True)
class MonatomicParticleState:
    spin_angular_momentum: GenericSpin  # the sole angular momentum
    displacement: ParticleDisplacement = field(default_factory=ParticleDisplacement)
    parallel_velocity: float
    gyromagnetic_ratio: float = -2.04e8  # default value for 3He

    @property
    def spin(self) -> GenericSpin:
        return self.spin_angular_momentum

    def as_coherent(self) -> list[CoherentMonatomicParticleState]:
        """Convert to a CoherentMonatomicParticleState."""
        return [
            CoherentMonatomicParticleState(
                spin_angular_momentum=s.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
                gyromagnetic_ratio=self.gyromagnetic_ratio,
            )
            for s in self.spin.flat_iter()
        ]


@dataclass(kw_only=True, frozen=True)
class CoherentMonatomicParticleState(MonatomicParticleState):
    """Represents the state of a coherent particle in the simulation."""

    def __post_init__(self) -> None:
        """Ensure that the spin is a CoherentSpin."""
        assert self.spin.size == 1, (
            "CoherentMonatomicParticleState must represent a single coherent spin."
        )
