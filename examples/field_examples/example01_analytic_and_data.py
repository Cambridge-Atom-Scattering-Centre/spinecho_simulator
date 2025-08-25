from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion
from spinecho_sim.util import make_linear_bz_data


def main() -> None:
    # Analytic region: on-axis Bz(z) = 2 T for 0 <= z <= 1
    analytic = FieldRegion.analytic(
        bz=lambda z: 2.0 if 0.0 <= z <= 1.0 else 0.0, length=1.0, z_start=0.0, dz=1e-5
    )

    # Data region: Bz linearly drops 2 T -> 0 T over z ∈ [1.0, 1.5]
    data = make_linear_bz_data(1.0, 1.5, 2.0, 0.0)

    # Scalar samples
    print("Analytic @ (0,0,0.5):", analytic.field_at(0.0, 0.0, 0.5))
    print("Data     @ (0,0,1.2):", data.field_at(0.0, 0.0, 1.2))

    # Vectorized samples on a small grid
    xyz = np.array(
        [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75], [0.0, 0.0, 1.25], [0.05, 0.05, 1.25]]
    )
    print("analytic.field_at_many:\n", analytic.field_at_many(xyz))
    print("data.field_at_many:\n", data.field_at_many(xyz))

    # Extents & membership
    print("Analytic extent:", analytic.extent)
    print("Data extent:", data.extent)
    print("Analytic contains (0,0,0.5):", analytic.contains(0.0, 0.0, 0.5))
    print("Data contains (0,0,0.5):", data.contains(0.0, 0.0, 0.5))


if __name__ == "__main__":
    main()
