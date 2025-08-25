from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion, UniformFieldRegion, ZeroField
from spinecho_sim.util import make_bx_blob


def main() -> None:
    # Base analytic uniform 1 T over [0, 1]
    base = FieldRegion.analytic(bz=lambda z: 1.0 if 0 <= z <= 1 else 0.0, length=1.0)

    # Localized Bx blob (data)
    blob = make_bx_blob(x_half=0.1, y_half=0.1, z0=0.0, z1=1.0, amplitude=0.1)

    # Uniform 0.5 T background
    uniform = UniformFieldRegion(B=np.array([0.0, 0.0, 0.5]), region_extent=base.extent)

    # Superpose two ways
    total_a = base + blob + uniform
    total_b = sum([base, blob, uniform], start=ZeroField())

    q = (0.05, 0.0, 0.5)
    print("base @", q, "=", base.field_at(*q))
    print("blob @", q, "=", blob.field_at(*q))
    print("uniform @", q, "=", uniform.field_at(*q))
    print("total_a @", q, "=", total_a.field_at(*q))
    print("total_b @", q, "=", total_b.field_at(*q))

    xyz = np.array([[0.05, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, 0.75]])
    print("total_a field_at_many:\n", total_a.field_at_many(xyz))

    print("total extent:", total_a.extent)
    print("total contains (0,0,0.5):", total_a.contains(0.0, 0.0, 0.5))


if __name__ == "__main__":
    main()
