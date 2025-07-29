from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload, override

import numpy as np


@dataclass(kw_only=True, frozen=True)
class ParticleDisplacement:
    """Represents the displacement of a particle in the simulation.

    This defines the displacement in the x-y plane, perpendicular to the particle's velocity.
    The displacement is stored in polar coordinates (r, theta), where:
    - r is the radial distance from the origin (0, 0) in the x-y plane.
    - theta is the angle from the positive x-axis in the x-y plane.
    """

    r: float = 0
    theta: float = 0

    @property
    def x(self) -> float:
        """Get the x-coordinate of the displacement."""
        return self.r * np.cos(self.theta)

    @property
    def y(self) -> float:
        """Get the y-coordinate of the displacement."""
        return self.r * np.sin(self.theta)

    @staticmethod
    def from_cartesian(x: float, y: float) -> ParticleDisplacement:
        """Create a ParticleDisplacement from Cartesian coordinates."""
        return ParticleDisplacement(r=np.sqrt(x**2 + y**2), theta=np.arctan2(y, x))


@dataclass(kw_only=True, frozen=True)
class ParticleDisplacementList(Sequence[ParticleDisplacement]):
    """A list of particle displacements."""

    r: np.ndarray[Any, np.dtype[np.floating]]
    theta: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if self.theta.shape != self.r.shape:
            msg = "Theta and r arrays must have the same shape."
            raise ValueError(msg)

    @staticmethod
    def from_displacements(
        displacements: Iterable[ParticleDisplacement],
    ) -> ParticleDisplacementList:
        """Create a ParticleDisplacementList from a list of ParticleDisplacements."""
        displacements = list(displacements)
        return ParticleDisplacementList(
            r=np.array([d.r for d in displacements]),
            theta=np.array([d.theta for d in displacements]),
        )

    @override
    def __len__(self) -> int:
        return self.theta.shape[0]

    @overload
    def __getitem__(self, index: int) -> ParticleDisplacement: ...

    @overload
    def __getitem__(self, index: slice) -> ParticleDisplacementList: ...

    @override
    def __getitem__(
        self, index: int | slice
    ) -> ParticleDisplacement | ParticleDisplacementList:
        if isinstance(index, slice):
            return ParticleDisplacementList(r=self.r[index], theta=self.theta[index])
        return ParticleDisplacement(r=self.r.item(index), theta=self.theta.item(index))

    @override
    def __iter__(self) -> Iterator[ParticleDisplacement]:
        for r, t in zip(self.r, self.theta, strict=True):
            yield ParticleDisplacement(r=r, theta=t)

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the x-displacement of the particles."""
        return np.array([displacement.x for displacement in self])

    @property
    def y(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the y-displacement of the particles."""
        return np.array([displacement.y for displacement in self])

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the displacement list."""
        return self.r.shape
