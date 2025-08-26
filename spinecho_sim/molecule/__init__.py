"""Module implementing simulation of the Ramsey homonuclear diatomic molecule."""

from __future__ import annotations

from spinecho_sim.molecule.hamiltonian_dicke import (
    build_diatomic_hamiltonian_dicke,
)
from spinecho_sim.molecule.hamiltonian_majorana import (
    build_diatomic_hamiltonian_majorana,
)

__all__ = [
    "build_diatomic_hamiltonian_dicke",
    "build_diatomic_hamiltonian_majorana",
]
