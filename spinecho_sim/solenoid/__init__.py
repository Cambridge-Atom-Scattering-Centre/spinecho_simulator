"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.solenoid._plot import (
    plot_expectation_angles,
    plot_expectation_trajectories,
    plot_expectation_trajectory,
    plot_expectation_values,
    plot_spin_states,
)
from spinecho_sim.solenoid._solenoid import (
    DiatomicSolenoid,
    DiatomicSolenoidSimulationResult,
    DiatomicSolenoidTrajectory,
    MonatomicSolenoid,
    MonatomicSolenoidSimulationResult,
    MonatomicSolenoidTrajectory,
)

__all__ = [
    "DiatomicSolenoid",
    "DiatomicSolenoidSimulationResult",
    "DiatomicSolenoidTrajectory",
    "MonatomicSolenoid",
    "MonatomicSolenoidSimulationResult",
    "MonatomicSolenoidTrajectory",
    "plot_expectation_angles",
    "plot_expectation_trajectories",
    "plot_expectation_trajectory",
    "plot_expectation_values",
    "plot_spin_states",
]
