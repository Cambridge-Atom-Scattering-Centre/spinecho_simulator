from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import starmap
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload, override

import numpy as np
from scipy.interpolate import (  # pyright: ignore[reportMissingTypeStubs]
    RegularGridInterpolator,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class AABB(NamedTuple):  # Axis-Aligned Bounding Box
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


@dataclass(kw_only=True)
class FieldRegion(ABC):
    """Abstract base class for a magnetic field region."""

    @abstractmethod
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        """Compute the (Bx, By, Bz) field at coordinates (x, y, z)."""
        ...

    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # default: scalar loop; override in data regions
        out = np.empty_like(xyz)
        for i, (x, y, z) in enumerate(xyz):
            out[i, :] = self.field_at(x, y, z)
        return out

    @property
    def extent(self) -> AABB | None:
        return None  # “unbounded” by default

    def contains(self, x: float, y: float, z: float) -> bool:
        if self.extent is None:
            return True
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.extent
        return (xmin <= x <= xmax) and (ymin <= y <= ymax) and (zmin <= z <= zmax)


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
    dz: float = 1e-5  # Step size for numerical derivatives if needed

    @property
    @override
    def extent(self) -> AABB:
        """Override the extent property to define the region's bounding box."""
        return AABB(
            (-np.inf, np.inf),  # x-range
            (-np.inf, np.inf),  # y-range
            (self.z_start, self.z_start + self.length),  # z-range
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
            # Use central difference to approximate derivative
            b0_p = (self.Bz_axis(z + self.dz) - self.Bz_axis(z - self.dz)) / (
                2 * self.dz
            )
        # Second derivative for Bz (numeric if no analytic derivative provided)
        if self.Bz_axis_second_deriv:
            b0_pp = self.Bz_axis_second_deriv(z)
        else:
            b0_pp = (self.Bz_axis(z + self.dz) - 2 * b0 + self.Bz_axis(z - self.dz)) / (
                self.dz**2
            )

        # Compute off-axis components using paraxial expansion
        b_r = -0.5 * r * b0_p
        b_z_off = b0 + (-0.25 * r**2 * b0_pp)

        # Resolve Br into x and y components
        b_x = b_r * (x / r)
        b_y = b_r * (y / r)
        return np.array([b_x, b_y, b_z_off])

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Mask for points within the region
        in_bounds = (xyz[:, 2] >= self.z_start) & (
            xyz[:, 2] <= self.z_start + self.length
        )

        # Compute fields for in-bounds points
        if np.any(in_bounds):
            x_in = xyz[in_bounds, 0]
            y_in = xyz[in_bounds, 1]
            z_in = xyz[in_bounds, 2]

            # Compute on-axis field and derivatives
            b0 = np.vectorize(self.Bz_axis)(z_in)
            if self.Bz_axis_deriv:
                b0_p = np.vectorize(self.Bz_axis_deriv)(z_in)
            else:
                dz = 1e-5
                b0_p = (
                    np.vectorize(self.Bz_axis)(z_in + dz)
                    - np.vectorize(self.Bz_axis)(z_in - dz)
                ) / (2 * dz)

            if self.Bz_axis_second_deriv:
                b0_pp = np.vectorize(self.Bz_axis_second_deriv)(z_in)
            else:
                dz = 1e-5
                b0_pp = (
                    np.vectorize(self.Bz_axis)(z_in + dz)
                    - 2 * b0
                    + np.vectorize(self.Bz_axis)(z_in - dz)
                ) / (dz**2)

            # Radial distance in x-y plane
            r = np.hypot(x_in, y_in)

            # Compute off-axis components using paraxial expansion
            b_r = -0.5 * r * b0_p
            b_z_off = b0 + (-0.25 * r**2 * b0_pp)

            # Resolve Br into x and y components
            b_x = np.zeros_like(r)
            b_y = np.zeros_like(r)
            nonzero_r = r > 0
            b_x[nonzero_r] = b_r[nonzero_r] * (x_in[nonzero_r] / r[nonzero_r])
            b_y[nonzero_r] = b_r[nonzero_r] * (y_in[nonzero_r] / r[nonzero_r])

            # Assign results to the output array
            result[in_bounds, 0] = b_x
            result[in_bounds, 1] = b_y
            result[in_bounds, 2] = b_z_off

        return result


@overload
def make_rgi3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    *,
    method: Literal["linear", "nearest"] = "linear",
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
    method: Literal["linear", "nearest"] = "linear",
    bounds_error: bool = ...,
    fill_value: None = ...,
) -> RegularGridInterpolator: ...


def make_rgi3d(  # noqa: PLR0913
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray[tuple[int, int, int, int]],
    *,
    method: Literal["linear", "nearest"] = "linear",
    bounds_error: bool = False,
    fill_value: float | None = None,
) -> RegularGridInterpolator:
    # cast(fill_value, Any) sidesteps the stub that disallows None
    return RegularGridInterpolator(
        (x, y, z),
        values,
        method=method,
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
        # Ensure input arrays are contiguous and of type float64
        self.x_vals = np.asarray(self.x_vals, dtype=np.float64, order="C")
        self.y_vals = np.asarray(self.y_vals, dtype=np.float64, order="C")
        self.z_vals = np.asarray(self.z_vals, dtype=np.float64, order="C")
        self.field_data = np.asarray(self.field_data, dtype=np.float64, order="C")
        # (optional) sanity checks
        for arr, name in (
            (self.x_vals, "x_vals"),
            (self.y_vals, "y_vals"),
            (self.z_vals, "z_vals"),
        ):
            if np.any(np.diff(arr) <= 0):
                msg = f"{name} must be strictly increasing"
                raise ValueError(msg)

        # Validate field_data shape
        if self.field_data.shape[-1] != 3:  # noqa: PLR2004
            msg = "field_data must have last dimension = 3 (Bx,By,Bz)"
            raise ValueError(msg)

        interpolator = make_rgi3d(
            self.x_vals,
            self.y_vals,
            self.z_vals,
            self.field_data.astype(np.float64, copy=False),
            bounds_error=False,
            fill_value=0.0,  # disallow extrapolation
        )
        object.__setattr__(self, "_interpolator", interpolator)

    @property
    @override
    def extent(self) -> AABB:
        """Override the extent property to define the region's bounding box."""
        return AABB(
            (self.x_vals[0], self.x_vals[-1]),  # x-range
            (self.y_vals[0], self.y_vals[-1]),  # y-range
            (self.z_vals[0], self.z_vals[-1]),  # z-range
        )

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

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Check if any points are out of bounds
        x_min, x_max = self.x_vals[0], self.x_vals[-1]
        y_min, y_max = self.y_vals[0], self.y_vals[-1]
        z_min, z_max = self.z_vals[0], self.z_vals[-1]

        out_of_bounds = (
            (xyz[:, 0] < x_min)
            | (xyz[:, 0] > x_max)
            | (xyz[:, 1] < y_min)
            | (xyz[:, 1] > y_max)
            | (xyz[:, 2] < z_min)
            | (xyz[:, 2] > z_max)
        )

        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Compute fields for in-bounds points
        in_bounds = ~out_of_bounds
        if np.any(in_bounds):
            result[in_bounds] = self._interpolator(xyz[in_bounds])

        return result


@dataclass(kw_only=True)
class FieldSequence(FieldRegion):
    """Composite field region that concatenates multiple regions end-to-end along z."""

    regions: list[FieldRegion]

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        for region in self.regions:
            if region.contains(x, y, z):
                return region.field_at(x, y, z)
        # If no region covered this z, return zero field
        return np.array([0.0, 0.0, 0.0])

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Mask to track which points have been assigned a field
        unassigned = np.ones(xyz.shape[0], dtype=bool)

        # Iterate over regions and assign fields for in-bounds points
        for region in self.regions:
            if not np.any(unassigned):
                break  # All points have been assigned

            # Get in-bounds points for this region
            in_bounds = np.array(list(starmap(region.contains, xyz[unassigned])))

            # Compute fields for in-bounds points
            if np.any(in_bounds):
                indices = np.where(unassigned)[0][in_bounds]
                result[indices] = region.field_at_many(xyz[indices])
                unassigned[indices] = False  # Mark these points as assigned

        return result


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
            if region.contains(x, y, z):
                b_region = region.field_at(x, y, z)
                if b_region.shape != (3,):  # Ensure consistent shape
                    msg = f"Region {region} returned invalid shape {b_region.shape}"
                    raise ValueError(msg)
                b_total += b_region
        return b_total

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Sum the contributions from all regions
        for region in self.regions:
            result += region.field_at_many(xyz)

        return result


@dataclass(kw_only=True)
class RotatedFieldRegion(FieldRegion):
    """Field region that rotates another region about the z-axis by a given angle."""

    base_region: FieldRegion
    angle: float  # rotation angle in radians (positive rotation about z-axis)
    _to_base_rotation: np.ndarray = field(init=False, repr=False)
    _to_global_rotation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        c, s = np.cos(self.angle), np.sin(self.angle)
        self._to_global_rotation = np.array(
            [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        )  # rotate vector to global
        self._to_base_rotation = (
            self._to_global_rotation.T
        )  # rotate point to base (opposite angle)

    @override
    def field_at(
        self, x: float, y: float, z: float
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]:
        point_base = self._to_base_rotation @ np.array(
            [x, y, z]
        )  # Rotate point to base region's frame

        # Get field in base region's coordinate system
        b_base = self.base_region.field_at(
            *point_base
        )  # np.array([Bx_base, By_base, Bz])

        return (
            self._to_global_rotation @ b_base
        )  # Rotate field vector back to global frame

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Rotate points to the base region's frame
        points_base = xyz @ self._to_base_rotation.T

        # Query the base region for the rotated points
        fields_base = self.base_region.field_at_many(points_base)

        # Rotate the field vectors back to the global frame
        return fields_base @ self._to_global_rotation.T


@dataclass(kw_only=True)
class TranslatedFieldRegion(FieldRegion):
    base_region: FieldRegion
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    @override
    def field_at(self, x: float, y: float, z: float) -> np.ndarray:
        return self.base_region.field_at(x - self.dx, y - self.dy, z - self.dz)

    @override
    def field_at_many(
        self, xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]
    ) -> np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]:
        # Apply the translation to all points
        translated_xyz = xyz - np.array([self.dx, self.dy, self.dz])

        # Query the base region for the translated points
        return self.base_region.field_at_many(translated_xyz)

    @property
    @override
    def extent(self) -> AABB | None:
        e = self.base_region.extent
        if e is None:
            return None
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = e
        return AABB(
            (xmin + self.dx, xmax + self.dx),
            (ymin + self.dy, ymax + self.dy),
            (zmin + self.dz, zmax + self.dz),
        )
