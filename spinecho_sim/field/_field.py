from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import pairwise
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    cast,
    overload,
    override,
)

import numpy as np
import scipy.integrate  # pyright: ignore[reportMissingTypeStubs]
from scipy.interpolate import (  # pyright: ignore[reportMissingTypeStubs]
    RegularGridInterpolator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from spinecho_sim.util import Array3, Vec3


def integrate_quad_typed(func: Callable[[float], float], a: float, b: float) -> float:
    """Integrate a function using quadrature (adaptive Simpson's rule)."""
    return scipy.integrate.quad(func, a, b)[0]  # pyright: ignore[reportUnknownVariableType]


class AABB(NamedTuple):  # Axis-Aligned Bounding Box
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @override
    def __repr__(self) -> str:
        return (
            "Axis-Aligned Bounding Box:\n"
            f"(x=({float(self.x[0]):.1f}, {float(self.x[1]):.1f}), "
            f"y=({float(self.y[0]):.1f}, {float(self.y[1]):.1f}), "
            f"z=({float(self.z[0]):.1f}, {float(self.z[1]):.1f}))"
        )


@dataclass(kw_only=True, frozen=True)
class FieldRegion(ABC):
    """Abstract base class for a magnetic field region."""

    @abstractmethod
    def field_at(self, x: float, y: float, z: float) -> Vec3:
        """Compute the (Bx, By, Bz) field at coordinates (x, y, z)."""
        ...

    def field_at_many(self, xyz: Array3) -> Array3:
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

    def contains_many(self, xyz: Array3) -> np.ndarray:
        e = self.extent
        if e is None:
            return np.ones(xyz.shape[0], dtype=bool)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = e
        return (
            (xmin <= xyz[:, 0])
            & (xyz[:, 0] <= xmax)
            & (ymin <= xyz[:, 1])
            & (xyz[:, 1] <= ymax)
            & (zmin <= xyz[:, 2])
            & (xyz[:, 2] <= zmax)
        )

    @classmethod
    def analytic(  # noqa: PLR0913
        cls,
        *,
        bz: Callable[[float], float],
        length: float,
        z_start: float = 0.0,
        bz_deriv: Callable[[float], float] | None = None,
        bz_second_deriv: Callable[[float], float] | None = None,
        dz: float = 1e-5,
    ) -> AnalyticFieldRegion:
        return AnalyticFieldRegion(
            bz_axis=bz,
            length=length,
            z_start=z_start,
            bz_axis_deriv=bz_deriv,
            bz_axis_second_deriv=bz_second_deriv,
            dz=dz,
        )

    @classmethod
    def from_data(
        cls,
        *,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        z_vals: np.ndarray,
        field_data: np.ndarray,
    ) -> DataFieldRegion:
        """Create a DataFieldRegion from explicit keyword arguments."""
        # Pass validated arguments to the DataFieldRegion constructor
        return DataFieldRegion(
            x_vals=x_vals,
            y_vals=y_vals,
            z_vals=z_vals,
            field_data=field_data,
        )

    # --- Composition Operators ---
    def __add__(self, other: FieldRegion) -> FieldSuperposition:
        """Combine two regions by superposing their fields."""
        return FieldSuperposition(regions=[self, other])

    def then(self, other: FieldRegion) -> FieldSequence:
        """Combine two regions sequentially along the z-axis."""
        return FieldSequence(regions=[self, other])

    def __radd__(self, other: object) -> FieldRegion:
        """Support sum([...], start=ZeroField())."""
        if isinstance(other, ZeroField) or other == 0:
            return self
        if isinstance(other, FieldRegion):
            return other + self
        return NotImplemented

    def rotate(self, angle: float) -> RotatedFieldRegion:
        """Rotate the field by a constant angle (in radians)."""
        return RotatedFieldRegion(base_region=self, angle=angle)

    def scale(self, factor: float) -> ScaledFieldRegion:
        """Scale the field by a constant factor."""
        return ScaledFieldRegion(base_region=self, factor=factor)

    def translate(
        self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
    ) -> TranslatedFieldRegion:
        """Translate the field by a constant offset."""
        return TranslatedFieldRegion(base_region=self, dx=dx, dy=dy, dz=dz)


@dataclass(kw_only=True, frozen=True)
class ZeroField(FieldRegion):
    """A field region that always returns zero field."""

    @override
    def field_at(self, x: float, y: float, z: float) -> Vec3:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
        return np.zeros_like(xyz, dtype=np.float64)


@dataclass(kw_only=True, frozen=True)
class UniformFieldRegion(FieldRegion):
    B: np.ndarray  # shape (3,)
    region_extent: AABB | None = None  # Internal storage for extent

    @property
    @override
    def extent(self) -> AABB | None:
        """Override the extent property to define the region's bounding box."""
        return self.region_extent

    @override
    def field_at(self, x: float, y: float, z: float) -> np.ndarray:
        return self.B if self.contains(x, y, z) else np.zeros(3)


@dataclass(kw_only=True, frozen=True)
class AnalyticFieldRegion(FieldRegion):
    """Analytic field region defined by an on-axis Bz(z) profile (axisymmetric)."""

    bz_axis: Callable[[float], float]  # User-supplied on-axis Bz(z) function
    length: float  # Length of this region along z-axis
    z_start: float = 0.0  # Starting z-coordinate of this region
    bz_axis_deriv: Callable[[float], float] | None = None  # Optional derivative of Bz
    bz_axis_second_deriv: Callable[[float], float] | None = (
        None  # Optional second derivative of Bz
    )
    dz: float = 1e-5  # Step size for numerical derivatives if needed

    _bz_v: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _bz_p_v: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _bz_pp_v: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Bind vectorized callables once
        bz = self.bz_axis
        dz = self.dz
        object.__setattr__(self, "_bz_v", np.vectorize(bz, otypes=[float]))
        if self.bz_axis_deriv:
            object.__setattr__(
                self, "_bz_p_v", np.vectorize(self.bz_axis_deriv, otypes=[float])
            )
        else:
            # Define a regular function with type annotations
            def numerical_derivative(z: np.ndarray) -> np.ndarray:
                return (self._bz_v(z + dz) - self._bz_v(z - dz)) / (2 * dz)

            object.__setattr__(
                self,
                "_bz_p_v",
                numerical_derivative,
            )
        if self.bz_axis_second_deriv:
            object.__setattr__(
                self,
                "_bz_pp_v",
                np.vectorize(self.bz_axis_second_deriv, otypes=[float]),
            )
        else:
            # Define a regular function with type annotations
            def numerical_derivative(z: np.ndarray) -> np.ndarray:
                return (self._bz_v(z + dz) - 2 * self._bz_v(z) + self._bz_v(z - dz)) / (
                    dz * dz
                )

            object.__setattr__(
                self,
                "_bz_pp_v",
                numerical_derivative,
            )

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
    def field_at(self, x: float, y: float, z: float) -> Vec3:
        # If z is outside this region, return zero field
        if not (self.z_start <= z <= self.z_start + self.length):
            return np.asarray([0.0, 0.0, 0.0])

        # Compute on-axis field and derivatives at this z
        b0 = self.bz_axis(z)  # on-axis Bz

        # Radial distance in x-y plane
        r = np.hypot(x, y)
        epsilon = 1e-15  # Small threshold for numerical stability
        if r < epsilon:
            return np.array([0.0, 0.0, b0])

        # First derivative of Bz (numeric if no analytic derivative provided)
        if self.bz_axis_deriv:
            b0_p = self.bz_axis_deriv(z)
        else:
            # Use central difference to approximate derivative
            b0_p = (self.bz_axis(z + self.dz) - self.bz_axis(z - self.dz)) / (
                2 * self.dz
            )
        # Second derivative for Bz (numeric if no analytic derivative provided)
        if self.bz_axis_second_deriv:
            b0_pp = self.bz_axis_second_deriv(z)
        else:
            b0_pp = (self.bz_axis(z + self.dz) - 2 * b0 + self.bz_axis(z - self.dz)) / (
                self.dz**2
            )

        # Compute off-axis components using paraxial expansion
        b_r = -0.5 * r * b0_p
        b_z_off = b0 - 0.25 * r**2 * b0_pp

        # Resolve Br into x and y components
        b_x = b_r * (x / r)
        b_y = b_r * (y / r)
        return np.array([b_x, b_y, b_z_off])

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
        # Initialize output array
        out = np.zeros_like(xyz, dtype=np.float64)

        # Mask for points within the region
        m = self.contains_many(xyz)  # use the vectorized contains
        if not np.any(m):
            return out

        # Extract x, y, z for in-bounds points
        x, y, z = xyz[m, 0], xyz[m, 1], xyz[m, 2]
        r = np.hypot(x, y)

        # Compute field components
        b0 = self._bz_v(z)
        b0_p = self._bz_p_v(z)
        b0_pp = self._bz_pp_v(z)
        epsilon = 1e-15  # Small threshold for numerical stability
        br = -0.5 * r * b0_p
        bz = b0 - 0.25 * (r * r) * b0_pp

        # Safely compute bx and by
        with np.errstate(divide="ignore", invalid="ignore"):
            bx = np.zeros_like(br)
            by = np.zeros_like(br)
            nonzero_r = r > epsilon
            bx[nonzero_r] = br[nonzero_r] * (x[nonzero_r] / r[nonzero_r])
            by[nonzero_r] = br[nonzero_r] * (y[nonzero_r] / r[nonzero_r])

        # Assign computed values to the output array
        out[m, 0], out[m, 1], out[m, 2] = bx, by, bz
        return out


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


@dataclass(kw_only=True, frozen=True)
class DataFieldRegion(FieldRegion):
    """Field region defined by discrete data on a 3D grid."""

    x_vals: np.ndarray  # 1D array of grid coordinates in x
    y_vals: np.ndarray  # 1D array of grid coordinates in y
    z_vals: np.ndarray  # 1D array of grid coordinates in z
    field_data: np.ndarray[
        tuple[int, int, int, int]
    ]  # 4D array of shape (Nx, Ny, Nz, 3) with Bx,By,Bz

    _interpolator: RegularGridInterpolator = field(init=False, repr=False)

    def validate(self) -> None:
        # Check for NaNs or infinite values in field_data
        if not np.isfinite(self.field_data).all():
            msg = "field_data contains NaN or infinite values."
            raise ValueError(msg)

        # Check for uniform spacing in x_vals, y_vals, and z_vals
        for arr, name in (
            (self.x_vals, "x_vals"),
            (self.y_vals, "y_vals"),
            (self.z_vals, "z_vals"),
        ):
            diffs = np.diff(arr)
            if not np.allclose(diffs, diffs[0]):
                msg = f"{name} must have uniform spacing."
                raise ValueError(msg)

    def __post_init__(self) -> None:
        # Ensure input arrays are contiguous and of type float64
        object.__setattr__(
            self, "x_vals", np.asarray(self.x_vals, dtype=np.float64, order="C")
        )
        object.__setattr__(
            self, "y_vals", np.asarray(self.y_vals, dtype=np.float64, order="C")
        )
        object.__setattr__(
            self, "z_vals", np.asarray(self.z_vals, dtype=np.float64, order="C")
        )
        object.__setattr__(
            self, "field_data", np.asarray(self.field_data, dtype=np.float64, order="C")
        )
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
        self.validate()

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
    def field_at(self, x: float, y: float, z: float) -> Vec3:
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
            "Vec3",
            self._interpolator(point).flatten(),
        )  # shape (3,)

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
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


@dataclass(kw_only=True, frozen=True)
class FieldSequence(FieldRegion):
    """Composite field region that concatenates multiple regions end-to-end along z."""

    regions: list[FieldRegion]

    def __post_init__(self) -> None:
        # Flatten nested FieldSequence instances
        flattened_regions = []
        for region in self.regions:
            if isinstance(region, FieldSequence):
                flattened_regions.extend(region.regions)
            else:
                flattened_regions.append(region)
        object.__setattr__(self, "regions", flattened_regions)

        # Validate non-overlapping z-spans
        z_spans = [
            (region.extent.z[0], region.extent.z[1])
            for region in self.regions
            if region.extent is not None
        ]
        z_spans.sort()  # Sort by start of z-span
        for (z1_start, z1_end), (z2_start, z2_end) in pairwise(z_spans):
            if z1_end > z2_start:
                msg = f"Overlapping z-spans detected: ({z1_start}, {z1_end}) and ({z2_start}, {z2_end})"
                raise ValueError(msg)

    @override
    def field_at(self, x: float, y: float, z: float) -> Vec3:
        for region in self.regions:
            if region.contains(x, y, z):
                return region.field_at(x, y, z)
        # If no region covered this z, return zero field
        return np.array([0.0, 0.0, 0.0])

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Mask to track which points have been assigned a field
        unassigned = np.ones(xyz.shape[0], dtype=bool)

        # Iterate over regions and assign fields for in-bounds points
        for region in self.regions:
            if not np.any(unassigned):
                break  # All points have been assigned
            m = unassigned & region.contains_many(xyz)
            if m.any():
                result[m] = region.field_at_many(xyz[m])
                unassigned[m] = False

        return result

    @property
    @override
    def extent(self) -> AABB | None:
        """Compute the union of extents of all regions."""
        extents = [
            region.extent for region in self.regions if region.extent is not None
        ]
        if not extents:
            return None  # Unbounded if no regions have extents

        x_min = min(e.x[0] for e in extents)
        x_max = max(e.x[1] for e in extents)
        y_min = min(e.y[0] for e in extents)
        y_max = max(e.y[1] for e in extents)
        z_min = min(e.z[0] for e in extents)
        z_max = max(e.z[1] for e in extents)
        return AABB((x_min, x_max), (y_min, y_max), (z_min, z_max))

    @override
    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if any region contains the point."""
        return any(region.contains(x, y, z) for region in self.regions)


@dataclass(kw_only=True, frozen=True)
class FieldSuperposition(FieldRegion):
    """Composite field region that superposes multiple regions (sums their fields)."""

    regions: list[FieldRegion]

    def __post_init__(self) -> None:
        # Flatten nested FieldSuperposition instances
        flattened_regions = []
        for region in self.regions:
            if isinstance(region, FieldSuperposition):
                flattened_regions.extend(region.regions)
            else:
                flattened_regions.append(region)
        object.__setattr__(self, "regions", flattened_regions)

    @override
    def field_at(self, x: float, y: float, z: float) -> Vec3:
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
    def field_at_many(self, xyz: Array3) -> Array3:
        # Initialize output array
        result = np.zeros_like(xyz, dtype=np.float64)

        # Sum the contributions from all regions
        for region in self.regions:
            result += region.field_at_many(xyz)

        return result

    @property
    @override
    def extent(self) -> AABB | None:
        """Compute the union of extents of all regions."""
        extents = [
            region.extent for region in self.regions if region.extent is not None
        ]
        if not extents:
            return None  # Unbounded if no regions have extents

        x_min = min(e.x[0] for e in extents)
        x_max = max(e.x[1] for e in extents)
        y_min = min(e.y[0] for e in extents)
        y_max = max(e.y[1] for e in extents)
        z_min = min(e.z[0] for e in extents)
        z_max = max(e.z[1] for e in extents)
        return AABB((x_min, x_max), (y_min, y_max), (z_min, z_max))

    @override
    def contains(self, x: float, y: float, z: float) -> bool:
        """Check if any region contains the point."""
        return any(region.contains(x, y, z) for region in self.regions)


@dataclass(kw_only=True, frozen=True)
class RotatedFieldRegion(FieldRegion):
    """Field region that rotates another region about the z-axis by a given angle."""

    base_region: FieldRegion
    angle: float  # rotation angle in radians (positive rotation about z-axis)
    _to_base_rotation: np.ndarray = field(init=False, repr=False)
    _to_global_rotation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        c, s = np.cos(self.angle), np.sin(self.angle)
        object.__setattr__(
            self, "_to_global_rotation", np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        )  # rotate vector to global
        object.__setattr__(
            self, "_to_base_rotation", self._to_global_rotation.T
        )  # rotate point to base (opposite angle)

    @property
    @override
    def extent(self) -> AABB | None:
        e = self.base_region.extent
        if e is None:
            return None
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = e
        # corners in xy
        corners = np.array(
            [[xmin, ymin, 0], [xmin, ymax, 0], [xmax, ymin, 0], [xmax, ymax, 0]]
        )
        rot = self._to_global_rotation
        rc = corners @ rot.T
        x_min, y_min = rc[:, 0].min(), rc[:, 1].min()
        x_max, y_max = rc[:, 0].max(), rc[:, 1].max()
        return AABB((x_min, x_max), (y_min, y_max), (zmin, zmax))

    @override
    def field_at(self, x: float, y: float, z: float) -> Vec3:
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
    def field_at_many(self, xyz: Array3) -> Array3:
        # Rotate points to the base region's frame
        points_base = xyz @ self._to_base_rotation.T

        # Query the base region for the rotated points
        fields_base = self.base_region.field_at_many(points_base)

        # Rotate the field vectors back to the global frame
        return fields_base @ self._to_global_rotation.T


@dataclass(kw_only=True, frozen=True)
class TranslatedFieldRegion(FieldRegion):
    base_region: FieldRegion
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    @override
    def field_at(self, x: float, y: float, z: float) -> np.ndarray:
        return self.base_region.field_at(x - self.dx, y - self.dy, z - self.dz)

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
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


@dataclass(kw_only=True, frozen=True)
class ScaledFieldRegion(FieldRegion):
    base_region: FieldRegion
    factor: float

    @override
    def field_at(self, x: float, y: float, z: float) -> Vec3:
        return self.base_region.field_at(x, y, z) * self.factor

    @override
    def field_at_many(self, xyz: Array3) -> Array3:
        return self.base_region.field_at_many(xyz) * self.factor

    @property
    @override
    def extent(self) -> AABB | None:
        return self.base_region.extent


@dataclass(kw_only=True, frozen=True)
class SolenoidRegion(AnalyticFieldRegion):
    """Ergonomic builders that mirror the old Solenoid constructors."""

    @classmethod
    def with_uniform_z(
        cls, *, length: float, strength: float, z_start: float = 0.0
    ) -> SolenoidRegion:
        """Bz is constant; AnalyticFieldRegion enforces 0 outside z-span."""
        return cls(bz_axis=lambda _z: strength, length=length, z_start=z_start)

    @classmethod
    def with_nonuniform_z(
        cls, *, length: float, strength: Callable[[float], float], z_start: float = 0.0
    ) -> SolenoidRegion:
        """Build a solenoid with a non-uniform field along the z-axis."""
        return cls(bz_axis=strength, length=length, z_start=z_start)

    @classmethod
    def from_experimental_parameters(
        cls,
        *,
        length: float,
        magnetic_constant: float,
        current: float,
        z_start: float = 0.0,
    ) -> SolenoidRegion:
        """Build a solenoid from an experimental magnetic constant and current."""
        amp = np.pi * magnetic_constant * current / (2 * length)

        def shape(z: float) -> float:
            return np.sin(np.pi * (z - z_start) / length) ** 2

        integral_bz = integrate_quad_typed(shape, z_start, z_start + length)
        print(integral_bz)

        amp = magnetic_constant * current / integral_bz

        def bz(z: float) -> float:
            return amp * np.sin(np.pi * (z - z_start) / length) ** 2

        return cls(bz_axis=bz, length=length, z_start=z_start)
