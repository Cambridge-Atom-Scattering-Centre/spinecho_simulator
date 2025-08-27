from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
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


@dataclass
class PlotConfig:
    cmap: str = "viridis"
    figsize: tuple[float, float] = (12, 8)
    line_styles: dict[str, str] | None = None
    show_magnitude: bool = True
    show_colorbar: bool = True
    title: str = "Magnetic Field Along z-Axis"
    save_path: str | None = None


@dataclass
class HeatmapConfig:
    cmap: str = "viridis"
    figsize: tuple[float, float] = (12, 8)
    title: str | None = None
    show_colorbar: bool = True
    show_contours: bool = False
    n_contours: int = 10
    show_field_lines: bool = False
    field_line_density: float = 1.0
    field_line_color: str = "w"
    field_line_width: float = 0.5
    symmetric_scale: bool = False
    save_path: str | None = None


DEFAULT_PLOT_CONFIG = PlotConfig()
DEFAULT_HEATMAP_CONFIG = HeatmapConfig()  # Module-level singleton variable


def determine_z_range(
    field_region: FieldRegion, z_start: float | None, z_end: float | None
) -> tuple[float, float]:
    """Determine the z-axis range for the heatmap."""
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


def create_field_plot(
    z_values: np.ndarray,
    fields: np.ndarray,
    *,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> tuple[Figure, Axes]:
    """Create a plot of the magnetic field."""
    # Default line styles
    default_styles = {"Bx": "r-", "By": "g-", "Bz": "b-", "magnitude": "k--"}

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
    z_start: float | None = None,
    z_end: float | None = None,
    num_points: int = 1000,
    *,
    config: PlotConfig = DEFAULT_PLOT_CONFIG,
) -> tuple[Figure, Axes]:
    """Plot magnetic field components along the z-axis (at x=0, y=0)."""
    z_start, z_end = determine_z_range(field_region, z_start, z_end)

    # Create points along z-axis
    z_values = np.linspace(z_start, z_end, num_points)
    points = cast("Array3", np.zeros((num_points, 3)))
    points[:, 2] = z_values

    # Calculate field at each point
    fields = field_region.field_at_many(points)

    # Create plot
    fig, ax = create_field_plot(z_values, fields, config=config)

    ax.set_xlabel("z position (m)")
    ax.set_ylabel("Field (T)")
    ax.set_title(config.title)
    ax.legend()
    ax.grid(visible=True)
    ax.set_xlim(z_start, z_end)

    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def create_heatmap(  # noqa: PLR0913, PLR0917
    component_values: np.ndarray,
    x_max: float,
    z_range: tuple[float, float],
    title: str | None,
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
    if title:
        ax.set_title(title)
    return fig, ax, im


def create_contours(  # noqa: PLR0913, PLR0917
    ax: Axes,
    xz_meshes: tuple[np.ndarray, np.ndarray],
    field_region: FieldRegion,
    component_values: np.ndarray,
    x_max: float,
    z_range: tuple[float, float],
    config: HeatmapConfig,
) -> None:
    # Add contour lines if requested
    if config.show_contours:
        contour_levels = np.linspace(
            float(component_values.min()),
            float(component_values.max()),
            config.n_contours,
        )
        ax.contour(
            xz_meshes[0],
            xz_meshes[1],
            component_values,
            levels=contour_levels,
            colors="k",
            alpha=0.5,
            linewidths=0.5,
        )

    # Add field lines if requested
    if config.show_field_lines:
        # Create a grid for the streamplot
        nx_stream, nz_stream = 50, 50  # Smaller grid for streamplot
        x_stream = np.linspace(0, x_max, nx_stream)
        z_stream = np.linspace(z_range[0], z_range[1], nz_stream)
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


def compute_field_component(
    field_region: FieldRegion,
    component: Literal["Bx", "By", "Bz", "magnitude"],
    x_max: float,
    z_range: tuple[float, float],
    n_xz: tuple[int, int],
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """Compute the field component values for the heatmap."""
    # Create 2D grid of points
    x_values = np.linspace(0, x_max, n_xz[0])  # Only positive x (radius)
    z_values = np.linspace(z_range[0], z_range[1], n_xz[1])

    x_mesh, z_mesh = np.meshgrid(x_values, z_values)
    points = cast("Array3", np.zeros((n_xz[0] * n_xz[1], 3)))
    points[:, 0] = x_mesh.flatten()  # x coordinates
    points[:, 2] = z_mesh.flatten()  # z coordinates

    # Calculate field at each point
    fields = field_region.field_at_many(points)

    # Reshape and extract the desired component
    fields_reshaped = fields.reshape((n_xz[1], n_xz[0], 3))

    if component == "Bx":
        component_values = fields_reshaped[:, :, 0]
        component_label = "B_x (T)"
    elif component == "By":
        component_values = fields_reshaped[:, :, 1]
        component_label = "B_y (T)"
    elif component == "Bz":
        component_values = fields_reshaped[:, :, 2]
        component_label = "B_z (T)"
    elif component == "magnitude":
        component_values = np.linalg.norm(fields_reshaped, axis=2)
        component_label = "|B| (T)"
    return component_values, component_label, x_mesh, z_mesh


def plot_field_heatmap(  # noqa: PLR0913, PLR0917
    field_region: FieldRegion,
    component: Literal["Bx", "By", "Bz", "magnitude"] = "Bz",
    x_max: float = 1.16e-3,  # Beam Radius
    z_start: float | None = None,
    z_end: float | None = None,
    n_xz: tuple[int, int] = (100, 500),
    config: HeatmapConfig = DEFAULT_HEATMAP_CONFIG,
) -> tuple[Figure, Axes]:
    """Create a heatmap of the magnetic field in the x-z plane (at y=0)."""
    z_start, z_end = determine_z_range(field_region, z_start, z_end)
    component_values, component_label, x_mesh, z_mesh = compute_field_component(
        field_region, component, x_max, (z_start, z_end), n_xz
    )
    # Set up color mapping
    if config.symmetric_scale and component != "magnitude":
        vmax = np.max(np.abs(component_values))
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        norm = None

    fig, ax, im = create_heatmap(
        component_values, x_max, (z_start, z_end), config.title, norm, config
    )

    create_contours(
        ax,
        (x_mesh, z_mesh),
        field_region,
        component_values,
        x_max,
        (z_start, z_end),
        config,
    )

    # Set title and labels
    if config.title is None:
        config.title = f"Magnetic Field {component} in X-Z Plane (Y=0)"
    ax.set_title(config.title)
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Z position (m)")
    ax.set_xlim(0, x_max)
    ax.set_ylim(z_start, z_end)

    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches="tight")

    # Add colorbar
    if config.show_colorbar:
        color_bar = fig.colorbar(im, ax=ax)
        color_bar.set_label(component_label)
    return fig, ax
