from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from spinecho_sim.state import ParticleDisplacement

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class FieldRegion(ABC):
    """
    Represents a contiguous z-interval on which an analytic / tabulated formula is valid.

    The region knows its own (z_min, z_max) and can
    evaluate the vector field at any (r,φ,z) *inside* that interval.
    """

    z_min: float
    z_max: float

    # convenience: allow region(z) to return B⃗ on-axis (r=0)
    def __call__(self, z: float, r: float = 0.0) -> np.ndarray:
        return self.b(z, ParticleDisplacement(r=r, theta=0.0))

    @abstractmethod
    def b(
        self, z: float, displacement: ParticleDisplacement
    ) -> np.ndarray:  # r,φ optional → pass xyz =(x,y,z)
        """Return magnetic-field vector at *xyz* (Tesla)."""
        ...

    # chaining helper
    def shift(self, dz: float) -> FieldRegion:
        """Return a *copy* translated by +dz in z."""
        return dataclasses.replace(self, z_min=self.z_min + dz, z_max=self.z_max + dz)


@dataclass
class CosSolenoid(FieldRegion):
    """
    Simplest analytic model:   Bz(r=0,z) = B0 * cos(k z)   for z∈[z_min,z_max].

    Off-axis field: first-order expansion in (r/R) *or* full on-axis derivative
    if you have it in closed form.
    """

    B0: float  # Tesla
    length: float  # m
    z_min: float = 0.0
    z_max: float = 1.0

    def _b_on_axis(self, z: float) -> float:
        return self.B0 * np.sin(np.pi * (z - self.z_min) / self.length) ** 2

    @override
    def b(self, z: float, displacement: ParticleDisplacement) -> np.ndarray:
        if not (self.z_min <= z <= self.z_max):
            return np.zeros(3)  # outside region → “no field”
        # Paraxial (r≪R) expansion:   Bz(r,z) ≈ Bz(0,z) * (1 - (r**2)/(2R**2))
        bz0 = self._b_on_axis(z)
        bz = bz0 * (1.0 - 0.5 * (displacement.r) ** 2)
        return np.array([0.0, 0.0, bz])


@dataclass
class TabulatedField(FieldRegion):
    """
    Represents a generic 1-D field profile Bz(0,z) given as sample points.

    Off-axis field: first-order expansion in (r/R) *or* full on-axis derivative
    if you have it in closed form.
    """

    z_grid: np.ndarray  # (N,)  mono-increasing
    Bz_grid: np.ndarray  # (N,)  same units
    z_min: float = field(init=False)
    z_max: float = field(init=False)

    _spline: CubicSpline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.z_min = float(self.z_grid[0])
        self.z_max = float(self.z_grid[-1])
        self._spline = CubicSpline(self.z_grid, self.Bz_grid, extrapolate=False)

    @override
    def b(self, z: float, displacement: ParticleDisplacement) -> np.ndarray:
        if not (self.z_min <= z <= self.z_max):
            return np.zeros(3)
        bz0 = float(self._spline(z - self.z_min))
        bz = bz0 * (1.0 - 0.5 * (displacement.r) ** 2)
        return np.array([0.0, 0.0, bz])


@dataclass
class PiecewiseField(FieldRegion):
    regions: Sequence[FieldRegion]

    z_min: float = field(init=False)
    z_max: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.regions:
            msg = "PiecewiseField requires at least one region."
            raise ValueError(msg)
        self.regions = tuple(sorted(self.regions, key=lambda r: r.z_min))
        self.z_min = self.regions[0].z_min
        self.z_max = self.regions[-1].z_max

    @override
    def b(self, z: float, displacement: ParticleDisplacement) -> np.ndarray:
        for r in self.regions:
            if r.z_min <= z <= r.z_max:
                return r.b(z, displacement)
        msg = f"z={z} is outside all regions"
        raise ValueError(msg)


@dataclass
class SuperposedField(FieldRegion):
    regions: Sequence[FieldRegion]

    z_min: float = field(init=False)
    z_max: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.regions:
            msg = "SuperposedField requires at least one region."
            raise ValueError(msg)
        self.z_min = min(r.z_min for r in self.regions)
        self.z_max = max(r.z_max for r in self.regions)

    @override
    def b(self, z: float, displacement: ParticleDisplacement) -> np.ndarray:
        return sum((r.b(z, displacement) for r in self.regions), np.zeros(3))
