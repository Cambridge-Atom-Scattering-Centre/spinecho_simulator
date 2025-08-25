from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion, ZeroField


def main() -> None:
    # Base analytic uniform 1 T over [0,1]
    base = FieldRegion.analytic(bz=lambda z: 1.0 if 0 <= z <= 1 else 0.0, length=1.0)

    # Localized "dipole-like" Bx blob from data
    x = np.linspace(-0.1, 0.1, 11)
    y = np.linspace(-0.1, 0.1, 11)
    z = np.linspace(0.0, 1.0, 11)
    vals = np.zeros((len(x), len(y), len(z), 3))
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            for iz, zz in enumerate(z):
                bx = (
                    0.1
                    * np.exp(-(((zz - 0.5) / 0.2) ** 2))
                    * np.exp(-((xx**2 + yy**2) / 0.01))
                )
                vals[ix, iy, iz, :] = [bx, 0.0, 0.0]
    blob = FieldRegion.from_data(x_vals=x, y_vals=y, z_vals=z, field_data=vals)

    # Superpose two ways:
    total_a = base + blob
    total_b = sum([base, blob], start=ZeroField())  # thanks to __radd__ support

    q = (0.05, 0.0, 0.5)
    print("total_a @", q, "=", total_a.field_at(*q))
    print("total_b @", q, "=", total_b.field_at(*q))

    # Vectorized
    xyz = np.array([[0.05, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, 0.75]])
    print("total_a many:\n", total_a.field_at_many(xyz))


if __name__ == "__main__":
    main()
