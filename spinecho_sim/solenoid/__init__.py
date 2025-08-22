"""Module for simulating and plotting solenoid magnetic fields and particle trajectories."""

from __future__ import annotations

from spinecho_sim.solenoid._diatomic_plot import (
    plot_diatomic_alignment_diagnostics,
    plot_diatomic_alignment_tensor,
    plot_diatomic_expectation_differences,
    plot_diatomic_expectation_values,
    plot_diatomic_normalisation,
)
from spinecho_sim.solenoid._plot import (
    animate_diatomic_mean_expectation_vectors,
    animate_monatomic_mean_expectation_vectors,
    plot_monatomic_expectation_angles,
    plot_monatomic_expectation_trajectories,
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
    "animate_diatomic_mean_expectation_vectors",
    "animate_monatomic_mean_expectation_vectors",
    "plot_diatomic_alignment_diagnostics",
    "plot_diatomic_alignment_tensor",
    "plot_diatomic_expectation_differences",
    "plot_diatomic_expectation_values",
    "plot_diatomic_normalisation",
    "plot_monatomic_expectation_angles",
    "plot_monatomic_expectation_trajectories",
    "plot_monatomic_expectation_values",
    "plot_monatomic_spin_states",
]
