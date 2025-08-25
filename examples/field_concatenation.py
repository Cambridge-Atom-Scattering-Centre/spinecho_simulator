from __future__ import annotations

import numpy as np

from spinecho_sim.field import AnalyticFieldRegion, DataFieldRegion


def main() -> None:
    """Demonstrates the functionality of different FieldRegion implementations and their concatenation using FieldSequence."""

    # Define an analytic field: uniform 2.0 T on-axis from z=0 to z=1, zero outside
    def solenoid_func(z: float) -> float:
        """Define the on-axis Bz(z) profile for an analytic solenoid field."""
        return 2.0 if 0.0 <= z <= 1.0 else 0.0

    # Create an AnalyticFieldRegion for the solenoid
    region1 = AnalyticFieldRegion(bz_axis=solenoid_func, length=1.0, z_start=0.0)

    # Create coordinate arrays for a small grid from z=1.0 to z=1.5 (next 0.5 m)
    x_vals = np.linspace(-0.1, 0.1, 5)  # 0.1 m radius grid in x
    y_vals = np.linspace(-0.1, 0.1, 5)
    z_vals = np.linspace(1.0, 1.5, 6)  # z from 1.0 to 1.5

    # Create dummy field data: linearly decreasing Bz from 2.0 T to 0 T in Bz
    field_data = np.zeros((len(x_vals), len(y_vals), len(z_vals), 3))
    for ix, _x in enumerate(x_vals):
        for iy, _y in enumerate(y_vals):
            for iz, zz in enumerate(z_vals):
                # Example field: linearly decreasing Bz from 2.0 at z=1.0 to 0 at z=1.5
                b_z_val = 2.0 * max(0.0, 1.5 - zz) / 0.5
                field_data[ix, iy, iz, :] = [0.0, 0.0, b_z_val]

    # Create a DataFieldRegion for the grid-based field
    region2 = DataFieldRegion(
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, field_data=field_data
    )

    # Concatenate region1 and region2 end-to-end
    combined_field = region1.then(region2)

    # --- Tests and Demonstrations ---
    print("=== Testing AnalyticFieldRegion ===")
    print(
        "Field at (0.0, 0.0, 0.5):", region1.field_at(0.0, 0.0, 0.5)
    )  # Inside region1
    print(
        "Field at (0.0, 0.0, 1.2):", region1.field_at(0.0, 0.0, 1.2)
    )  # Outside region1

    print("\n=== Testing DataFieldRegion ===")
    print(
        "Field at (0.0, 0.0, 1.2):", region2.field_at(0.0, 0.0, 1.2)
    )  # Inside region2
    print(
        "Field at (0.0, 0.0, 0.5):", region2.field_at(0.0, 0.0, 0.5)
    )  # Outside region2

    print("\n=== Testing FieldSequence ===")
    print(
        "Field at (0.0, 0.0, 0.5):", combined_field.field_at(0.0, 0.0, 0.5)
    )  # region1
    print(
        "Field at (0.0, 0.0, 1.2):", combined_field.field_at(0.0, 0.0, 1.2)
    )  # region2
    print(
        "Field at (0.0, 0.0, 1.6):", combined_field.field_at(0.0, 0.0, 1.6)
    )  # Outside all

    # --- Additional Tests ---
    print("\n=== Additional Tests ===")
    print("Field at (0.05, 0.05, 0.5):", combined_field.field_at(0.05, 0.05, 0.5))
    print("Field at (-0.05, -0.05, 1.2):", combined_field.field_at(-0.05, -0.05, 1.2))
    print(
        "Field at (0.1, 0.1, 1.6):", combined_field.field_at(0.1, 0.1, 1.6)
    )  # Outside

    # --- Edge Cases ---
    print("\n=== Edge Cases ===")
    print(
        "Field at (0.0, 0.0, 0.0):", combined_field.field_at(0.0, 0.0, 0.0)
    )  # Start of region1
    print(
        "Field at (0.0, 0.0, 1.5):", combined_field.field_at(0.0, 0.0, 1.5)
    )  # End of region2


if __name__ == "__main__":
    main()
