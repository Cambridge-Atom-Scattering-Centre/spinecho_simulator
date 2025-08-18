"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.solenoid._diatomic_plot import (
    plot_diatomic_expectation_values,
)
from spinecho_sim.solenoid._monatomic_plot import (
    plot_monatomic_expectation_angles,
    plot_monatomic_expectation_trajectories,
    plot_monatomic_expectation_trajectory,
    plot_monatomic_expectation_values,
    plot_monatomic_spin_states,
)
from spinecho_sim.solenoid._solenoid import (
    MonatomicSolenoid,
    MonatomicSolenoidSimulationResult,
    MonatomicSolenoidTrajectory,
    Solenoid,
    SolenoidSimulationResult,
    SolenoidTrajectory,
)

__all__ = [
    "MonatomicSolenoid",
    "MonatomicSolenoidSimulationResult",
    "MonatomicSolenoidTrajectory",
    "Solenoid",
    "SolenoidSimulationResult",
    "SolenoidTrajectory",
    "plot_diatomic_expectation_values",
    "plot_monatomic_expectation_angles",
    "plot_monatomic_expectation_trajectories",
    "plot_monatomic_expectation_trajectory",
    "plot_monatomic_expectation_values",
    "plot_monatomic_spin_states",
]
