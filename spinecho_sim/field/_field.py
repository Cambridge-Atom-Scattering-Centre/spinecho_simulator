from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

import numpy as np
from scipy.interpolate import (  # pyright: ignore[reportMissingTypeStubs]
    RegularGridInterpolator,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(kw_only=True)
class FieldRegion(ABC):
    """Abstract base class for a magnetic field region."""

    @abstractmethod
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        """Compute the (Bx, By, Bz) field at coordinates (x, y, z)."""
        ...


@dataclass(kw_only=True)
class AnalyticFieldRegion(FieldRegion):
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
            return np.asarray([0.0, 0.0, 0.0])

        # Compute on-axis field and derivatives at this z
        b0 = self.Bz_axis(z)  # on-axis Bz

        # Radial distance in x-y plane
        r = np.hypot(x, y)
        if r == 0:
            return np.array([0.0, 0.0, b0])

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

        # Compute off-axis components using paraxial expansion
        b_r = -0.5 * r * b0_p
        b_z_off = b0 + (-0.25 * r**2 * b0_pp)

        # Resolve Br into x and y components
        b_x = b_r * (x / r)
        b_y = b_r * (y / r)
        return np.array([b_x, b_y, b_z_off])


@overload
def make_rgi3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    *,
    bounds_error: bool = ...,
    fill_value: float = ...,
) -> RegularGridInterpolator: ...
@overload
def make_rgi3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    *,
    bounds_error: bool = ...,
    fill_value: None = ...,
) -> RegularGridInterpolator: ...


def make_rgi3d(  # noqa: PLR0913
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray[tuple[int, int, int, int]],
    *,
    bounds_error: bool = False,
    fill_value: float | None = None,
) -> RegularGridInterpolator:
    # cast(fill_value, Any) sidesteps the stub that disallows None
    return RegularGridInterpolator(
        (x, y, z),
        values,
        bounds_error=bounds_error,
        fill_value=cast("Any", fill_value),
    )


@dataclass(kw_only=True)
class DataFieldRegion(FieldRegion):
    """Field region defined by discrete data on a 3D grid."""

    x_vals: np.ndarray  # 1D array of grid coordinates in x
    y_vals: np.ndarray  # 1D array of grid coordinates in y
    z_vals: np.ndarray  # 1D array of grid coordinates in z
    field_data: np.ndarray[
        tuple[int, int, int, int]
    ]  # 4D array of shape (Nx, Ny, Nz, 3) with Bx,By,Bz

    _interpolator: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # (optional) sanity checks
        for arr, name in (
            (self.x_vals, "x_vals"),
            (self.y_vals, "y_vals"),
            (self.z_vals, "z_vals"),
        ):
            if np.any(np.diff(arr) <= 0):
                msg = f"{name} must be strictly increasing"
                raise ValueError(msg)
        if self.field_data.shape[-1] != 3:  # noqa: PLR2004
            msg = "field_data must have last dimension = 3 (Bx,By,Bz)"
            raise ValueError(msg)

        interpolator = make_rgi3d(
            self.x_vals,
            self.y_vals,
            self.z_vals,
            self.field_data.astype(np.float64, copy=False),
            bounds_error=False,
            fill_value=np.nan,  # disallow extrapolation
        )
        object.__setattr__(self, "_interpolator", interpolator)

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        # Check if the point is within the bounds of the grid
        out_of_bounds = (
            x < self.x_vals[0]
            or x > self.x_vals[-1]
            or y < self.y_vals[0]
            or y > self.y_vals[-1]
            or z < self.z_vals[0]
            or z > self.z_vals[-1]
        )
        if out_of_bounds:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Use the interpolator to get Bx,By,Bz at the given point
        point = np.array([x, y, z])
        # If point is outside provided grid range, fill_value=None will extrapolate;
        # Alternatively, we could choose to return zeros outside region:
        # if x < self.x_vals[0] or x > self.x_vals[-1] or ... (similar for y,z): return [0,0,0]
        return cast(
            "np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]",
            self._interpolator(point).flatten(),
        )  # shape (3,)


@dataclass(kw_only=True)
class FieldSequence(FieldRegion):
    """Composite field region that concatenates multiple regions end-to-end along z."""

    regions: list[FieldRegion]

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        for region in self.regions:
            # Here we assume each region knows its z-span and returns zero outside it
            b = region.field_at(x, y, z)
            # if not zero, then (x,y,z) fell inside this region's span
            if np.any(b != 0.0):
                return b
        # If no region covered this z, return zero field
        return np.array([0.0, 0.0, 0.0])


@dataclass(kw_only=True)
class FieldSuperposition(FieldRegion):
    """Composite field region that superposes multiple regions (sums their fields)."""

    regions: list[FieldRegion]

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        b_total = np.array([0.0, 0.0, 0.0])
        for region in self.regions:
            b_region = region.field_at(x, y, z)
            if b_region.shape != (3,):  # Ensure consistent shape
                msg = f"Region {region} returned invalid shape {b_region.shape}"
                raise ValueError(msg)
            b_total += b_region
        return b_total


@dataclass(kw_only=True)
class RotatedFieldRegion(FieldRegion):
    """Field region that rotates another region about the z-axis by a given angle."""

    base_region: FieldRegion
    angle: float  # rotation angle in radians (positive rotation about z-axis)

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        # Compute coordinates in the base region's frame by rotating point in opposite direction
        rotation_matrix = np.array(
            [
                [np.cos(-self.angle), -np.sin(-self.angle), 0],
                [np.sin(-self.angle), np.cos(-self.angle), 0],
                [0, 0, 1],
            ]
        )
        point = np.array([x, y, z])
        point_base = rotation_matrix @ point  # Rotate point to base region's frame

        # Get field in base region's coordinate system
        b_base = self.base_region.field_at(
            *point_base
        )  # np.array([Bx_base, By_base, Bz])

        # Rotate the field vector back to the global frame
        rotation_matrix_inv = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle), 0],
                [np.sin(self.angle), np.cos(self.angle), 0],
                [0, 0, 1],
            ]
        )
        return rotation_matrix_inv @ b_base  # Rotate field vector back to global frame
