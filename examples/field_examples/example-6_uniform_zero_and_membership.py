from __future__ import annotations

import numpy as np

from spinecho_sim.field import (
    FieldRegion,
    FieldSuperposition,
    UniformFieldRegion,
    ZeroField,
)


def main() -> None:
    uniform_all = UniformFieldRegion(B=np.array([0.0, 0.0, 0.1]))
    zero = ZeroField()

    # Two analytic pieces at different spans to show contains/extent
    field_a = FieldRegion.analytic(
        bz=lambda z: 1.0 if 0 <= z <= 0.5 else 0.0,  # noqa: PLR2004
        length=0.5,
    )
    field_b = FieldRegion.analytic(
        bz=lambda z: 0.5 if 0.7 <= z <= 1.2 else 0.0,  # noqa: PLR2004
        length=0.5,
        z_start=0.7,
    )

    sup = FieldSuperposition(regions=[uniform_all, field_a, field_b, zero])
    print("superposed extent:", sup.extent)

    xyz = np.array([[0, 0, 0.25], [0, 0, 0.6], [0, 0, 1.0], [0, 0, 2.0]])
    print("contains_many:", sup.contains_many(xyz))
    print("fields:\n", sup.field_at_many(xyz))


if __name__ == "__main__":
    main()
