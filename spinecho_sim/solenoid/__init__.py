"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.solenoid._plot import (
    plot_spin_angles,
    plot_spin_component,
    plot_spin_components,
    plot_spin_intensity,
    plot_spin_phi,
    plot_spin_theta,
)
from spinecho_sim.solenoid._solenoid import (
    Solenoid,
    SolenoidSimulationResult,
    SolenoidTrajectory,
)

__all__ = [
    "Solenoid",
    "SolenoidSimulationResult",
    "SolenoidTrajectory",
    "plot_spin_angles",
    "plot_spin_component",
    "plot_spin_components",
    "plot_spin_intensity",
    "plot_spin_phi",
    "plot_spin_theta",
]
