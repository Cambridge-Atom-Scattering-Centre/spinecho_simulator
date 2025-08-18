"""Module implementing simulation of the Ramsey homonuclear diatomic molecule."""

from __future__ import annotations

from spinecho_sim.molecule.hamiltonian_dicke import (
    diatomic_hamiltonian_dicke,
)
from spinecho_sim.molecule.hamiltonian_majorana import (
    diatomic_hamiltonian_majorana,
)

__all__ = [
    "diatomic_hamiltonian_dicke",
    "diatomic_hamiltonian_majorana",
]
