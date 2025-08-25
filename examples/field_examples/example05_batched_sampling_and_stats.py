from __future__ import annotations

import numpy as np

from spinecho_sim.field import FieldRegion


def main() -> None:
    # Analytic cos^2 on-axis profile over length L=1
    length = 1.0

    def bz(z: float) -> float:
        return (
            (np.cos(np.pi * (z - 0.5 * length) / length) ** 2)
            if 0 <= z <= length
            else 0.0
        )

    region = FieldRegion.analytic(bz=bz, length=length, z_start=0.0, dz=1e-6)

    # Sample cylindrical shell across z with vectorized call
    r = 0.05
    ntheta, nz = 128, 256
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    z = np.linspace(0.0, length, nz)
    x_mesh, z_mesh = np.meshgrid(np.cos(theta) * r, z, indexing="xy")
    y_mesh, _ = np.meshgrid(np.sin(theta) * r, z, indexing="xy")
    xyz = np.column_stack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()])

    b_field = region.field_at_many(xyz)  # (ntheta*nz, 3)
    b_mag = np.linalg.norm(b_field, axis=1).reshape(nz, ntheta)

    print("B magnitude stats @ r=0.05:")
    print(
        "min =",
        float(b_mag.min()),
        "max =",
        float(b_mag.max()),
        "mean =",
        float(b_mag.mean()),
    )


if __name__ == "__main__":
    main()
