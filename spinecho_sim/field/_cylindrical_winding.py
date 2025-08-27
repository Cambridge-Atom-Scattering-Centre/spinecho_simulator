from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.field import (
    AxisDataFieldRegion,
    FieldRegion,
    PlotConfig,
    plot_field_along_axis,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

MU0 = 4e-7 * np.pi  # [H/m]


@dataclass(frozen=True, kw_only=True)
class ProportionalPitchWinding:
    """Finite cylindrical winding with constant pitch across layers."""

    length: float
    radii: Sequence[float]  # layer radii [m]
    turns_per_layer: Sequence[int]  # turns per layer (finite)
    current_per_layer: Sequence[float]

    def _pitch(self) -> float:
        n_max = max(self.turns_per_layer) if self.turns_per_layer else 1
        return self.length / max(n_max, 1)

    @property
    def loop_lengths(self) -> np.ndarray:
        return self._pitch() * np.asarray(self.turns_per_layer)


def bz_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz(z) from a finite set of circular loops."""
    currents = np.asarray(winding.current_per_layer)
    radii = np.asarray(winding.radii)
    dz = z_vals[None, :] - winding.length / 2  # (L, M), centered about middle
    lengths = winding.loop_lengths

    denom_plus = np.sqrt(radii[:, None] ** 2 + (dz + 0.5 * lengths[:, None]) ** 2)
    denom_minus = np.sqrt(radii[:, None] ** 2 + (dz - 0.5 * lengths[:, None]) ** 2)
    contrib = (
        0.5
        * MU0
        * currents[:, None]
        * (
            (dz + 0.5 * lengths[:, None]) / denom_plus
            - (dz - 0.5 * lengths[:, None]) / denom_minus
        )
    )
    return contrib.sum(axis=0)  # (M,)


def bz_p_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz'(z) from a finite set of circular loops."""
    raise NotImplementedError
    current = np.asarray(winding.current_per_layer)
    radii = np.asarray(winding.radii)
    dz = z_vals[None, :] - winding.length / 2
    denom = (radii[:, None] ** 2 + dz**2) ** 2.5  # (a^2 + (z - z0)^2)^(5/2)
    contrib = (
        MU0 * current[:, None] * (radii[:, None] ** 2) * (-3.0 * dz) / (2.0 * denom)
    )
    return contrib.sum(axis=0)


def bz_pp_axis_from_loops(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
) -> np.ndarray:
    """Sum the on-axis field Bz''(z) from a finite set of circular loops."""
    raise NotImplementedError
    radii = np.asarray(winding.radii)
    current = np.asarray(winding.current_per_layer)
    dz = z_vals[None, :] - winding.length / 2
    r2 = radii[:, None] ** 2 + dz**2
    denom_7_2 = r2**3.5
    denom_5_2 = r2**2.5
    term = (-3.0) / denom_5_2 + (15.0 * dz**2) / denom_7_2
    contrib = MU0 * current[:, None] * (radii[:, None] ** 2) / 2.0 * term
    return contrib.sum(axis=0)


def make_axis_region_from_winding(
    z_vals: np.ndarray,
    winding: ProportionalPitchWinding,
    *,
    include_derivatives: bool = True,
) -> FieldRegion:
    """Compute on-axis data (and optional derivatives) for a finite set of turns laid out by 'winding', returns AxisDataFieldRegion."""
    bz = bz_axis_from_loops(
        z_vals,
        winding,
    )

    if include_derivatives:
        dbz = bz_p_axis_from_loops(z_vals, winding)
        d2bz = bz_pp_axis_from_loops(z_vals, winding)
        return AxisDataFieldRegion(
            z_vals=z_vals, bz_vals=bz, bz_deriv_vals=dbz, bz_second_deriv_vals=d2bz
        )

    return AxisDataFieldRegion(z_vals=z_vals, bz_vals=bz)


if __name__ == "__main__":
    # Axis sampling grid
    z = np.linspace(0.0, 0.75, 2001)

    # Three finite layers, small turn counts (discrete windings)
    w = ProportionalPitchWinding(
        length=0.75,
        radii=[1.65e-2, 2.295e-2],  # 16.5mm inner radius, 2.15mm coil radius
        turns_per_layer=[1370, 995],  # finite, discrete turns
        current_per_layer=[1.0, -1.0],
    )

    axis_region = make_axis_region_from_winding(z, w, include_derivatives=False)
    # -> hand 'axis_region' to your new AxisDataFieldRegion pipeline

    # Plot the field along the axis
    fig1, ax1 = plot_field_along_axis(
        axis_region,
        config=PlotConfig(
            title="On-Axis Field from Measured Data",
        ),
    )
    plt.show()
