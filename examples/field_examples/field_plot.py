from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim.field import (
    FieldSuperposition,
    HeatmapConfig,
    PlotConfig,
    SolenoidRegion,
    plot_field_along_axis,
    plot_field_heatmap,
)


def create_nested_solenoids(
    outer_length: float = 0.75,
    outer_current: float = 1.0,
    inner_length: float = 0.5,
    inner_current: float = -1.0,
    magnetic_constant: float = 3.96e-3,
) -> FieldSuperposition:
    """Create a field with one solenoid and a shorter one inside with reversed current."""
    # Center the inner solenoid within the outer one
    inner_offset = (outer_length - inner_length) / 2

    # Create the outer solenoid
    outer_solenoid = SolenoidRegion.from_experimental_parameters(
        length=outer_length,
        magnetic_constant=magnetic_constant,
        current=outer_current,
    )

    # Create the inner solenoid, centered inside the outer solenoid
    inner_solenoid = SolenoidRegion.from_experimental_parameters(
        length=inner_length,
        magnetic_constant=magnetic_constant,
        current=inner_current,
        z_start=inner_offset,
    )

    # Combine the two solenoids into a single field
    return outer_solenoid + inner_solenoid


if __name__ == "__main__":
    # Create the nested solenoids field
    field = create_nested_solenoids(
        outer_length=0.75,
        outer_current=1.0,
        inner_length=0.25,
        inner_current=-1.0,
        magnetic_constant=3.96e-3,
    )

    # Plot the field along the z-axis
    fig1, ax1 = plot_field_along_axis(
        field,
        config=PlotConfig(
            title="Nested Solenoids - Field Along Z-Axis",
            save_path="./examples/nested_solenoids_axial.png",
        ),
    )

    # Plot a heatmap of the Bz component
    fig2, ax2 = plot_field_heatmap(
        field,
        component="Bz",
        x_max=1.16e-3,  # Beam Radius
        config=HeatmapConfig(
            title="Nested Solenoids - Bz Component",
            cmap="coolwarm",
            symmetric_scale=True,
            show_field_lines=True,
            save_path="./examples/nested_solenoids_heatmap_bz.png",
        ),
    )

    # Plot a heatmap of the field magnitude
    fig3, ax3 = plot_field_heatmap(
        field,
        component="magnitude",
        x_max=1.16e-3,  # Beam Radius
        config=HeatmapConfig(
            title="Nested Solenoids - Field Magnitude",
            cmap="viridis",
            show_field_lines=True,
            save_path="./examples/nested_solenoids_heatmap_magnitude.png",
        ),
    )

    plt.show()
    print("Plots saved to examples/ directory")
