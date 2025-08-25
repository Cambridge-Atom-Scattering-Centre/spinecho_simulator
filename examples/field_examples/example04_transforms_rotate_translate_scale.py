from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.field import (
    FieldRegion,
    RotatedFieldRegion,
    ScaledFieldRegion,
    TranslatedFieldRegion,
)
from spinecho_sim.util import make_bx_blob


def visualize_extents(regions: list[FieldRegion], title: str = "Extents") -> None:
    _fig, ax = plt.subplots(figsize=(8, 6))
    for region in regions:
        e = region.extent
        if e is None:
            continue
        (xmin, xmax), (_ymin, _ymax), (zmin, zmax) = e
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [zmin, zmin, zmax, zmax, zmin],
            label=region.__class__.__name__,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.legend()
    plt.show()


def main() -> None:
    # Base blob (Bx)
    blob = make_bx_blob(x_half=1.0, y_half=1.0, z0=0.0, z1=1.0)

    # Transform chain: rotate (90°) -> translate (dz=0.3) -> scale (x=0.5)
    rotated = RotatedFieldRegion(base_region=blob, angle=np.pi / 2)
    translated = TranslatedFieldRegion(base_region=rotated, dz=0.3)
    scaled = ScaledFieldRegion(base_region=translated, scale=0.5)

    print("base extent:", blob.extent)
    print("rotated extent:", rotated.extent)
    print("translated extent:", translated.extent)
    print("scaled extent:", scaled.extent)

    # Sample vectorized
    pts = np.array([[0.4, 0.0, 0.5], [0.0, 0.4, 0.5], [0.4, 0.0, 0.9]])
    print("scaled.field_at_many:\n", scaled.field_at_many(pts))

    # Visualize extents of the transform chain
    visualize_extents(
        [blob, rotated, translated, scaled], title="Transform Chain Extents"
    )


if __name__ == "__main__":
    main()
