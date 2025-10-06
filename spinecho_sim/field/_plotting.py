from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

    from spinecho_sim.field._field import FieldRegion
    from spinecho_sim.util import Array3


class PlotLineStyles(TypedDict, total=False):
    Bx: str
    By: str
    Bz: str
    magnitude: str


@dataclass(kw_only=True, frozen=True)
class FieldPlotConfig:
    cmap: str = "viridis"
    figsize: tuple[float, float] = (12, 8)
    line_styles: PlotLineStyles | None = None
    show_magnitude: bool = True
    show_colorbar: bool = True


@dataclass(kw_only=True, frozen=True)
class HeatmapConfig:
    cmap: str = "viridis"
    figsize: tuple[float, float] = (12, 8)
    show_colorbar: bool = True
    n_contours: int | None = 10
    show_field_lines: bool = False
    field_line_density: float = 1.0
    field_line_color: str = "w"
    field_line_width: float = 0.5
    symmetric_scale: bool = False


DEFAULT_PLOT_CONFIG = FieldPlotConfig()
DEFAULT_HEATMAP_CONFIG = HeatmapConfig()  # Module-level singleton variable


def _determine_heatmap_z_range(
    field_region: FieldRegion, z_range: tuple[float | None, float | None]
) -> tuple[float, float]:
    z_start, z_end = z_range
    if z_start is None or z_end is None:
        extent = field_region.extent
        assert extent is not None, (
            "Field region has no extent and no z range was provided"
        )

        if z_start is None:
            z_start = extent.z[0]
        if z_end is None:
            z_end = extent.z[1]
    return z_start, z_end


def _create_field_plot(
    z_values: np.ndarray,
    fields: np.ndarray,
    *,
    config: FieldPlotConfig = DEFAULT_PLOT_CONFIG,
) -> tuple[Figure, Axes]:
    """Create a plot of the magnetic field."""
    # Default line styles
    default_styles: PlotLineStyles = {
        "Bx": "r-",
        "By": "g-",
        "Bz": "b-",
        "magnitude": "k--",
    }

    if config.line_styles:
        default_styles.update(config.line_styles)
    # Create plot
    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot field components
    ax.plot(z_values, fields[:, 0], default_styles["Bx"], label="Bx")
    ax.plot(z_values, fields[:, 1], default_styles["By"], label="By")
    ax.plot(z_values, fields[:, 2], default_styles["Bz"], label="Bz")

    if config.show_magnitude:
        magnitude = np.linalg.norm(fields, axis=1)
        ax.plot(z_values, magnitude, default_styles["magnitude"], label="|B|")
    return fig, ax


def plot_field_along_axis(
    field_region: FieldRegion,
    z_range: tuple[float | None, float | None] = (None, None),
    num_points: int = 1000,
    *,
    config: FieldPlotConfig = DEFAULT_PLOT_CONFIG,
) -> tuple[Figure, Axes]:
    """Plot magnetic field components along the z-axis (at x=0, y=0)."""
    z_start, z_end = _determine_heatmap_z_range(field_region, z_range)

    # Create points along z-axis
    z_values = np.linspace(z_start, z_end, num_points)
    points = cast("Array3", np.zeros((num_points, 3)))
    points[:, 2] = z_values

    # Calculate field at each point
    fields = field_region.field_at_many(points)

    # Create plot
    fig, ax = _create_field_plot(z_values, fields, config=config)

    ax.set_xlabel("z position (m)")
    ax.set_ylabel("Field (T)")
    ax.set_title("Magnetic Field Along z-Axis")
    ax.legend()
    ax.grid(visible=True)
    ax.set_xlim(z_start, z_end)

    return fig, ax


def create_heatmap(
    component_values: np.ndarray,
    x_max: float,
    z_range: tuple[float, float],
    norm: Normalize | None,
    config: HeatmapConfig,
) -> tuple[Figure, Axes, AxesImage]:
    """Create the heatmap plot."""
    fig, ax = plt.subplots(figsize=config.figsize)
    im = ax.imshow(
        component_values,
        aspect="auto",
        origin="lower",
        extent=(0, x_max, z_range[0], z_range[1]),
        cmap=config.cmap,
        norm=norm,
    )
    return fig, ax, im


def _create_contours(
    ax: Axes,
    x_mesh: np.ndarray,
    z_mesh: np.ndarray,
    component_values: np.ndarray,
    n_contours: int,
) -> None:
    contour_levels = np.linspace(
        float(component_values.min()), float(component_values.max()), n_contours
    )
    ax.contour(
        x_mesh,
        z_mesh,
        component_values,
        levels=contour_levels,
        colors="k",
        alpha=0.5,
        linewidths=0.5,
    )


def _create_field_lines(
    ax: Axes,
    x_mesh: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    z_mesh: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    field_region: FieldRegion,
    config: HeatmapConfig,
) -> None:
    x_max = float(x_mesh[0, -1])
    z_range = (float(z_mesh[0, 0]), float(z_mesh[-1, 0]))
    # Add field lines if requested
    if config.show_field_lines:
        # Create a grid for the streamplot
        nx_stream, nz_stream = 50, 50  # Smaller grid for streamplot
        x_stream = np.linspace(0, x_max, nx_stream)
        z_stream = np.linspace(*z_range, nz_stream)
        x_mesh, z_mesh = np.meshgrid(x_stream, z_stream)

        # Calculate field at each point in the streamplot grid
        points_stream = cast("Array3", np.zeros((nx_stream * nz_stream, 3)))
        points_stream[:, 0] = x_mesh.flatten()
        points_stream[:, 2] = z_mesh.flatten()

        fields_stream = field_region.field_at_many(points_stream)

        # Reshape to match grid
        b_x_stream = fields_stream[:, 0].reshape(nz_stream, nx_stream)
        b_z_stream = fields_stream[:, 2].reshape(nz_stream, nx_stream)

        # Plot streamlines
        ax.streamplot(
            x_stream,
            z_stream,
            b_x_stream,
            b_z_stream,
            density=config.field_line_density,
            color=config.field_line_color,
            linewidth=config.field_line_width,
            broken_streamlines=False,
        )


def _evaluate_field_region(
    field_region: FieldRegion,
    component: Literal["Bx", "By", "Bz", "magnitude"],
    x_mesh: np.ndarray,
    z_mesh: np.ndarray,
) -> np.ndarray:
    """Compute the field component values for the heatmap."""
    # Create 2D grid of points

    points = cast("Array3", np.zeros((x_mesh.size, 3)))
    points[:, 0] = x_mesh.flatten()  # x coordinates
    points[:, 2] = z_mesh.flatten()  # z coordinates

    # Calculate field at each point
    fields = field_region.field_at_many(points)

    # Reshape and extract the desired component
    fields_reshaped = fields.reshape((*x_mesh.shape, 3))

    if component == "Bx":
        component_values = fields_reshaped[:, :, 0]
    elif component == "By":
        component_values = fields_reshaped[:, :, 1]
    elif component == "Bz":
        component_values = fields_reshaped[:, :, 2]
    elif component == "magnitude":
        component_values = np.linalg.norm(fields_reshaped, axis=2)
    return component_values


def _get_field_component_label(
    component: Literal["Bx", "By", "Bz", "magnitude"],
) -> str:
    if component == "Bx":
        return "B_x (T)"
    if component == "By":
        return "B_y (T)"
    if component == "Bz":
        return "B_z (T)"
    if component == "magnitude":
        return "|B| (T)"
    msg = f"Unknown component: {component}"
    raise ValueError(msg)


def plot_field_heatmap(  # noqa: PLR0913, PLR0917
    field_region: FieldRegion,
    component: Literal["Bx", "By", "Bz", "magnitude"] = "Bz",
    x_max: float = 1.16e-3,  # Beam Radius
    z_range: tuple[float | None, float | None] = (None, None),
    n_x: int = 100,
    n_z: int = 500,
    config: HeatmapConfig = DEFAULT_HEATMAP_CONFIG,
) -> tuple[Figure, Axes]:
    """Create a heatmap of the magnetic field in the x-z plane (at y=0)."""
    z_start, z_end = _determine_heatmap_z_range(field_region, z_range)

    x_values = np.linspace(0, x_max, n_x)  # Only positive x (radius)
    z_values = np.linspace(z_start, z_end, n_z)
    x_mesh, z_mesh = np.meshgrid(x_values, z_values)

    component_values = _evaluate_field_region(field_region, component, x_mesh, z_mesh)
    component_label = _get_field_component_label(component)
    # Set up color mapping
    if config.symmetric_scale and component != "magnitude":
        vmax = np.max(np.abs(component_values))
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        norm = None

    fig, ax, im = create_heatmap(
        component_values, x_max, (z_start, z_end), norm, config
    )

    if config.n_contours is not None:
        _create_contours(ax, x_mesh, z_mesh, component_values, config.n_contours)
    if config.show_field_lines:
        _create_field_lines(ax, x_mesh, z_mesh, field_region, config)

    # Set title and labels
    ax.set_title(f"Magnetic Field {component} in X-Z Plane (Y=0)")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(z_start, z_end)

    # Add colorbar
    if config.show_colorbar:
        color_bar = fig.colorbar(im, ax=ax)
        color_bar.set_label(component_label)
    return fig, ax
