from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class FieldRegion(ABC):
    """Abstract base class for a magnetic field region."""

    @abstractmethod
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        """Compute the (Bx, By, Bz) field at coordinates (x, y, z)."""
        ...


@dataclass
class AnalyticSolenoidFieldRegion(FieldRegion):
    """Analytic field region defined by an on-axis Bz(z) profile (axisymmetric)."""

    Bz_axis: Callable[[float], float]  # User-supplied on-axis Bz(z) function
    length: float  # Length of this region along z-axis
    z_start: float = 0.0  # Starting z-coordinate of this region
    Bz_axis_deriv: Callable[[float], float] | None = None  # Optional derivative of Bz
    Bz_axis_second_deriv: Callable[[float], float] | None = (
        None  # Optional second derivative of Bz
    )

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        # If z is outside this region, return zero field
        if not (self.z_start <= z <= self.z_start + self.length):
            return np.array([0.0, 0.0, 0.0])

        # Compute on-axis field and derivatives at this z
        b0 = self.Bz_axis(z)  # on-axis Bz

        # First derivative of Bz (numeric if no analytic derivative provided)
        if self.Bz_axis_deriv:
            b0_p = self.Bz_axis_deriv(z)
        else:
            dz = 1e-5  # small step for numerical derivative
            # Use central difference to approximate derivative
            b0_p = (self.Bz_axis(z + dz) - self.Bz_axis(z - dz)) / (2 * dz)
        # Second derivative for Bz (numeric if no analytic derivative provided)
        if self.Bz_axis_second_deriv:
            b0_pp = self.Bz_axis_second_deriv(z)
        else:
            dz = 1e-5
            b0_pp = (self.Bz_axis(z + dz) - 2 * b0 + self.Bz_axis(z - dz)) / (dz**2)

        # Radial distance in x-y plane
        r = np.hypot(x, y)
        # Compute off-axis components using paraxial expansion
        b_r = -0.5 * r * b0_p
        b_z_off = b0 + (-0.25 * r**2 * b0_pp)

        # Resolve Br into x and y components
        if r != 0.0:
            b_x = b_r * (x / r)
            b_y = b_r * (y / r)
        else:
            b_x = b_y = 0.0
        return np.array([b_x, b_y, b_z_off])
