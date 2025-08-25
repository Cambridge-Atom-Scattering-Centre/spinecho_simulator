from __future__ import annotations

import numpy as np

from spinecho_sim.field import AnalyticFieldRegion, DataFieldRegion, FieldSuperposition

if __name__ == "__main__":
    # Analytic base field: e.g., 1 T uniform field in Z over 1 m
    def base_func(z: float) -> float:
        return 1.0 if 0.0 <= z <= 1.0 else 0.0

    base_region = AnalyticFieldRegion(Bz_axis=base_func, length=1.0, z_start=0.0)

    # Data field: e.g., a small dipole field centered at z=0.5 adding some Bx component
    x_vals = np.linspace(-0.1, 0.1, 11)
    y_vals = np.linspace(-0.1, 0.1, 11)
    z_vals = np.linspace(0.0, 1.0, 11)
    field_data = np.zeros((len(x_vals), len(y_vals), len(z_vals), 3))
    # Populate a simple dipole-like field: Bx = 0.1 T * (some profile), Bz = 0 (just as example)
    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            for iz, zz in enumerate(z_vals):
                # Example: horizontal field that decays with distance from center (0.5) and radius
                bx = (
                    0.1
                    * np.exp(-(((zz - 0.5) / 0.2) ** 2))
                    * np.exp(-((x**2 + y**2) / 0.01))
                )
                field_data[ix, iy, iz, :] = [bx, 0.0, 0.0]

    dipole_region = DataFieldRegion(
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, field_data=field_data
    )

    # Superpose the uniform solenoid field and the dipole field
    total_field = FieldSuperposition(regions=[base_region, dipole_region])

    # Query a point for demonstration
    B = total_field.field_at(0.05, 0.0, 0.5)
    print(B)  # This will include contributions from both base_region and dipole_region
