from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion, UniformFieldRegion
from spinecho_sim.util import make_linear_bz_data


def main() -> None:
    # Region A: Analytic 2 T in z ∈ [0, 1]
    region_a = FieldRegion.analytic(
        bz=lambda z: 2.0 if 0 <= z <= 1 else 0.0, length=1.0
    )

    # Region B: Data 2 T -> 0 T in z ∈ [1, 1.5]
    region_b = make_linear_bz_data(1.0, 1.5, 2.0, 0.0)

    # Region C: Uniform 1 T in z ∈ [1.5, 2.0]
    # (reusing an analytic region's extent for convenience)
    region_c_extent = FieldRegion.analytic(
        bz=lambda z: 1.0 if 1.5 <= z <= 2.0 else 0.0,  # noqa: PLR2004
        length=0.5,
        z_start=1.5,
    ).extent
    region_c = UniformFieldRegion(
        B=np.array([0.0, 0.0, 1.0]), region_extent=region_c_extent
    )

    seq = region_a.then(region_b).then(region_c)

    print("Sequence extent:", seq.extent)
    for zq in (0.5, 1.2, 1.6, 2.0):
        print(f"seq @ (0,0,{zq}):", seq.field_at(0.0, 0.0, zq))

    xyz = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.2], [0.0, 0.0, 1.6], [0.0, 0.0, 2.0]])
    print("seq.field_at_many:\n", seq.field_at_many(xyz))


if __name__ == "__main__":
    main()
