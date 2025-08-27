"""Module for flexible and extensible field representation framework."""

from __future__ import annotations

from spinecho_sim.field._field import (
    AnalyticFieldRegion,
    DataFieldRegion,
    FieldRegion,
    FieldSequence,
    FieldSuperposition,
    RotatedFieldRegion,
    ScaledFieldRegion,
    SolenoidRegion,
    TranslatedFieldRegion,
    UniformFieldRegion,
    ZeroField,
)
from spinecho_sim.field._plotting import (
    HeatmapConfig,
    PlotConfig,
    plot_field_along_axis,
    plot_field_heatmap,
)

__all__ = [
    "AnalyticFieldRegion",
    "DataFieldRegion",
    "FieldRegion",
    "FieldSequence",
    "FieldSuperposition",
    "HeatmapConfig",
    "PlotConfig",
    "RotatedFieldRegion",
    "ScaledFieldRegion",
    "SolenoidRegion",
    "TranslatedFieldRegion",
    "UniformFieldRegion",
    "ZeroField",
    "plot_field_along_axis",
    "plot_field_heatmap",
]
