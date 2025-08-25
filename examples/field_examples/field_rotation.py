from __future__ import annotations

import numpy as np

from spinecho_sim.field import DataFieldRegion, RotatedFieldRegion

if __name__ == "__main__":
    # Define a simple data field with a constant Bx field in some region (for example purposes)
    x_vals = np.linspace(-1.0, 1.0, 5)
    y_vals = np.linspace(-1.0, 1.0, 5)
    z_vals = np.linspace(0.0, 1.0, 3)
    field_data = np.zeros((len(x_vals), len(y_vals), len(z_vals), 3))
    field_data[..., 0] = 0.5  # Bx = 0.5 T everywhere in this grid, By=Bz=0
    base_region = DataFieldRegion(
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, field_data=field_data
    )

    # Create a rotated version of this region by 90 degrees about z (so Bx becomes By in global coords)
    rotated_region = RotatedFieldRegion(base_region=base_region, angle=np.pi / 2)

    # Compare fields at a point for base vs rotated
    B_base = base_region.field_at(0.5, 0.0, 0.5)  # should be [0.5, 0.0, 0.0]
    B_rot = rotated_region.field_at(
        0.5, 0.0, 0.5
    )  # base's (x=0.5,y=0) corresponds to rotated's (x=0, y=0.5)
    print("Base field:", B_base)  # e.g., [0.5 0.  0. ]
    print("Rotated field:", B_rot)  # e.g., [0.  0.5 0. ]
