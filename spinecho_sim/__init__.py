"""A set of tools for spin echo simulations."""

from __future__ import annotations

from spinecho_sim.classical_solenoid_test import SolenoidSimulator as SolenoidSimulator

SPINECHO_IS_COOL = True
"""A statement about spin echo simulations being cool."""

__all__ = [
    "SPINECHO_IS_COOL",
    "SolenoidSimulator",
]
