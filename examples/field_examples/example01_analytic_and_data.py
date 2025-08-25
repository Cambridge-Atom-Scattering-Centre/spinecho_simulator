from __future__ import annotations

import numpy as np

from spinecho_sim.field import (
    FieldRegion,
    RotatedFieldRegion,
    ScaledFieldRegion,
    TranslatedFieldRegion,
    UniformFieldRegion,
)


def main() -> None:
    # --- Analytic region: on-axis Bz(z) = 2.0 T for 0 <= z <= 1, else 0
    def bz(z: float) -> float:
        return 2.0 if 0.0 <= z <= 1.0 else 0.0

    analytic = FieldRegion.analytic(bz=bz, length=1.0, z_start=0.0, dz=1e-5)

    # --- Data region: simple Bz that linearly drops from 2 T to 0 T over z in [1.0, 1.5]
    x_vals = np.linspace(-0.1, 0.1, 5)
    y_vals = np.linspace(-0.1, 0.1, 5)
    z_vals = np.linspace(1.0, 1.5, 6)
    field_data = np.zeros((len(x_vals), len(y_vals), len(z_vals), 3))
    for ix, _x in enumerate(x_vals):
        for iy, _y in enumerate(y_vals):
            for iz, z in enumerate(z_vals):
                bz_val = 2.0 * max(0.0, 1.5 - z) / 0.5  # linear ramp down
                field_data[ix, iy, iz, :] = [0.0, 0.0, bz_val]

    data = FieldRegion.from_data(
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, field_data=field_data
    )

    # --- Uniform field region
    uniform = UniformFieldRegion(
        B=np.array([0.0, 0.0, 1.0]),
        region_extent=analytic.extent,
    )

    # --- Superposition of fields
    superposed = analytic + data + uniform

    # --- Sequential combination of fields
    sequential = analytic.then(data)

    # --- Transformed regions
    rotated = RotatedFieldRegion(base_region=analytic, angle=np.pi / 4)  # 45 degrees
    translated = TranslatedFieldRegion(base_region=data, dx=0.5, dy=0.5, dz=0.0)
    scaled = ScaledFieldRegion(base_region=uniform, scale=2.0)

    # --- Scalar samples
    print("Analytic @ (0,0,0.5):", analytic.field_at(0.0, 0.0, 0.5))  # inside
    print("Data     @ (0,0,1.2):", data.field_at(0.0, 0.0, 1.2))  # inside
    print("Uniform  @ (0,0,0.5):", uniform.field_at(0.0, 0.0, 0.5))  # uniform
    print("Superposed @ (0,0,0.5):", superposed.field_at(0.0, 0.0, 0.5))  # combined
    print("Sequential @ (0,0,1.2):", sequential.field_at(0.0, 0.0, 1.2))  # sequential
    print("Rotated @ (0.5,0.5,0.5):", rotated.field_at(0.5, 0.5, 0.5))  # rotated
    print("Translated @ (0.5,0.5,1.2):", translated.field_at(0.5, 0.5, 1.2))  # shifted
    print("Scaled @ (0,0,0.5):", scaled.field_at(0.0, 0.0, 0.5))  # scaled

    # --- Vectorized samples on a small grid
    xyz = np.array(
        [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75], [0.0, 0.0, 1.25], [0.05, 0.05, 1.25]]
    )
    print("analytic.field_at_many:\n", analytic.field_at_many(xyz))
    print("data.field_at_many:\n", data.field_at_many(xyz))
    print("superposed.field_at_many:\n", superposed.field_at_many(xyz))

    # --- Extents & membership
    print("Analytic extent:", analytic.extent)
    print("Data extent:", data.extent)
    print("Superposed extent:", superposed.extent)
    print("Analytic contains (0,0,0.5):", analytic.contains(0.0, 0.0, 0.5))
    print("Data contains (0,0,0.5):", data.contains(0.0, 0.0, 0.5))


if __name__ == "__main__":
    main()
