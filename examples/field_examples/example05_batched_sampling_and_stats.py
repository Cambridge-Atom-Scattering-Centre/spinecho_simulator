from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.field import FieldRegion, FieldSuperposition, UniformFieldRegion


def main() -> None:  # noqa: PLR0914
    length = 1.0

    def bz(z: float) -> float:
        return (
            (np.cos(np.pi * (z - 0.5 * length) / length) ** 2)
            if 0 <= z <= length
            else 0.0
        )

    analytic = FieldRegion.analytic(bz=bz, length=length, z_start=0.0, dz=1e-6)
    uniform = UniformFieldRegion(B=np.array([0.0, 0.0, 0.1]))
    region = FieldSuperposition(regions=[analytic, uniform])

    r = 0.05
    ntheta, nz = 128, 256
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    z = np.linspace(0.0, length, nz)
    x_mesh, z_mesh = np.meshgrid(np.cos(theta) * r, z, indexing="xy")
    y_mesh, _ = np.meshgrid(np.sin(theta) * r, z, indexing="xy")
    xyz = np.column_stack([x_mesh.ravel(), y_mesh.ravel(), z_mesh.ravel()])

    b_field = region.field_at_many(xyz)  # (nz*ntheta, 3)
    b_mag = np.linalg.norm(b_field, axis=1).reshape(nz, ntheta)
    print(
        "B magnitude stats @ r=0.05: min=",
        float(b_mag.min()),
        "max=",
        float(b_mag.max()),
        "mean=",
        float(b_mag.mean()),
    )

    visualize_field_magnitude(z, b_mag)


def visualize_field_magnitude(z: np.ndarray, b_mag: np.ndarray) -> None:
    b_mean = b_mag.mean(axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(z, b_mean, label="Mean |B|")
    plt.fill_between(
        z, b_mag.min(axis=1), b_mag.max(axis=1), alpha=0.2, label="Min/Max |B|"
    )
    plt.xlabel("z (m)")
    plt.ylabel("|B| (T)")
    plt.title("Magnetic Field Magnitude Along z")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
