from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, override

from spinecho_sim.state._displacement import ParticleDisplacement

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spinecho_sim.state._spin import GenericSpin


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseParticleState(ABC):
    """Data every species carries: trajectory *not* dynamics."""

    displacement: ParticleDisplacement = field(default_factory=ParticleDisplacement)
    parallel_velocity: float

    @property
    @abstractmethod
    def spins(self) -> tuple[GenericSpin, ...]: ...

    # ------ shared helper: expand into single-spin sub-states -------------
    @abstractmethod
    def as_coherent(self) -> Sequence[CoherentParticleState]: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class DiatomicParticleState(BaseParticleState):
    nuclear_angular_momentum: GenericSpin  # I
    rotation_angular_momentum: GenericSpin  # N

    @property
    @override
    def spins(self) -> tuple[GenericSpin, GenericSpin]:
        return (self.nuclear_angular_momentum, self.rotation_angular_momentum)

    @property
    def i(self) -> GenericSpin:
        return self.nuclear_angular_momentum

    @property
    def j(self) -> GenericSpin:
        return self.rotation_angular_momentum

    @override
    def as_coherent(self) -> Sequence[CoherentDiatomicParticleState]:
        return [
            CoherentDiatomicParticleState(
                nuclear_angular_momentum=nuc.as_generic(),
                rotation_angular_momentum=rot.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
            )
            for nuc, rot in product(
                self.nuclear_angular_momentum.flat_iter(),
                self.rotation_angular_momentum.flat_iter(),
            )
        ]


@dataclass(frozen=True, kw_only=True, slots=True)
class CoherentDiatomicParticleState(DiatomicParticleState):
    def __post_init__(self) -> None:
        assert self.nuclear_angular_momentum.size == 1
        assert self.rotation_angular_momentum.size == 1


@dataclass(frozen=True, kw_only=True, slots=True)
class MonatomicParticleState(BaseParticleState):
    spin_angular_momentum: GenericSpin
    gyromagnetic_ratio: float = -2.04e8  # default value for 3He

    @property
    @override
    def spins(self) -> tuple[GenericSpin,]:
        return (self.spin_angular_momentum,)

    @override
    def as_coherent(self) -> Sequence[CoherentMonatomicParticleState]:
        return [
            CoherentMonatomicParticleState(
                spin_angular_momentum=s.as_generic(),
                displacement=self.displacement,
                parallel_velocity=self.parallel_velocity,
                gyromagnetic_ratio=self.gyromagnetic_ratio,
            )
            for s in self.spin_angular_momentum.flat_iter()
        ]


@dataclass(frozen=True, kw_only=True, slots=True)
class CoherentMonatomicParticleState(MonatomicParticleState):
    def __post_init__(self) -> None:
        assert self.spin_angular_momentum.size == 1, (
            "CoherentParticleState must represent a single coherent spin."
        )


CoherentParticleState = CoherentMonatomicParticleState | CoherentDiatomicParticleState
