from __future__ import annotations

import numpy as np

from spinecho_sim.field import (
    DataFieldRegion,
    FieldRegion,
    RotatedFieldRegion,
    ScaledFieldRegion,
    TranslatedFieldRegion,
    UniformFieldRegion,
)


def make_blob_bx() -> DataFieldRegion:
    x = np.linspace(-1.0, 1.0, 21)
    y = np.linspace(-1.0, 1.0, 21)
    z = np.linspace(0.0, 1.0, 11)
    vals = np.zeros((len(x), len(y), len(z), 3))
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            for iz, zz in enumerate(z):
                vals[ix, iy, iz, 0] = np.exp(-((xx**2 + yy**2) / 0.2)) * np.exp(
                    -((zz - 0.5) ** 2) / 0.1
                )  # Bx-only blob
    return FieldRegion.from_data(x_vals=x, y_vals=y, z_vals=z, field_data=vals)


def main() -> None:
    blob = make_blob_bx()
    print("base extent:", blob.extent)

    # Rotate 90° about z -> Bx pattern becomes By in global frame
    rot = RotatedFieldRegion(base_region=blob, angle=np.pi / 2)
    print("rotated extent:", rot.extent)

    # Translate by +0.3 in z
    moved = TranslatedFieldRegion(base_region=rot, dz=0.3)
    print("translated extent:", moved.extent)

    # Scale amplitude by 0.5 (note: extent typically should remain geometric, not scaled)
    half = ScaledFieldRegion(base_region=moved, scale=0.5)
    print("scaled (amplitude) extent:", half.extent)

    # Add a uniform background Bz
    uniform = UniformFieldRegion(B=np.array([0.0, 0.0, 0.2]))

    total = half + uniform

    # Compare fields at a few points
    pts = np.array([[0.4, 0.0, 0.5], [0.0, 0.4, 0.5], [0.4, 0.0, 0.9]])
    print("total.field_at_many:\n", total.field_at_many(pts))


if __name__ == "__main__":
    main()
