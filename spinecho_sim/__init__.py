"""A set of tools for spin echo simulations."""

from __future__ import annotations

from spinecho_sim.solenoid import Solenoid, SolenoidSimulationResult, SolenoidTrajectory
from spinecho_sim.state import ParticleState

__all__ = [
    "ParticleState",
    "Solenoid",
    "SolenoidSimulationResult",
    "SolenoidTrajectory",
]
