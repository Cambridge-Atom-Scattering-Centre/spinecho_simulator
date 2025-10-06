from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.field import (
    AxisDataFieldRegion,
    HeatmapConfig,
    plot_field_along_axis,
    plot_field_heatmap,
)

if __name__ == "__main__":
    # Create some example on-axis field data
    z_vals = np.linspace(0, 0.75, 100)  # 100 points along z-axis from 0 to 0.75 meters
    # Example: sinusoidal field profile (similar to SolenoidRegion)
    bz_vals = np.sin(np.pi * z_vals / 0.75) ** 2

    # Create an AxisDataFieldRegion from this data
    field_region = AxisDataFieldRegion.from_measured_data(z_vals, bz_vals)

    # Plot the field along the axis
    fig1, ax1 = plot_field_along_axis(field_region)
    ax1.set_title("On-Axis Field from Measured Data")
    fig1.savefig("./examples/axis_data_field_axial.png")

    # Plot a heatmap of the field
    fig2, ax2 = plot_field_heatmap(
        field_region,
        component="Bz",
        x_max=0.05,  # 5 cm radius
        config=HeatmapConfig(
            cmap="coolwarm", symmetric_scale=True, show_field_lines=True
        ),
    )
    ax2.set_title("Field from Measured Data - Bz Component")
    fig2.savefig("./examples/axis_data_field_heatmap.png")

    plt.show()
    print("Plots saved to examples/ directory")
