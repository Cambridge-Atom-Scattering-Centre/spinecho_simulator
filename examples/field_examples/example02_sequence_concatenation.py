from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion


def main() -> None:
    # Region A: analytic, 2 T in z ∈ [0,1]
    field_a = FieldRegion.analytic(bz=lambda z: 2.0 if 0 <= z <= 1 else 0.0, length=1.0)

    # Region B: data in z ∈ [1,1.5], Bz linearly dropping to 0
    x = np.linspace(-0.1, 0.1, 5)
    y = np.linspace(-0.1, 0.1, 5)
    z = np.linspace(1.0, 1.5, 6)
    vals = np.zeros((len(x), len(y), len(z), 3))
    for ix, _xx in enumerate(x):
        for iy, _yy in enumerate(y):
            for iz, zz in enumerate(z):
                bz = 2.0 * max(0.0, 1.5 - zz) / 0.5
                vals[ix, iy, iz, :] = [0.0, 0.0, bz]
    field_b = FieldRegion.from_data(x_vals=x, y_vals=y, z_vals=z, field_data=vals)

    seq = field_a.then(field_b)

    print("Sequence extent:", seq.extent)
    for zq in (0.5, 1.2, 1.6):
        print(f"seq @ (0,0,{zq}):", seq.field_at(0.0, 0.0, zq))

    # Vectorized
    xyz = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.2], [0.0, 0.0, 1.6]])
    print("seq.field_at_many:\n", seq.field_at_many(xyz))


if __name__ == "__main__":
    main()
