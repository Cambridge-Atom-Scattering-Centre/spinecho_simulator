from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from spinecho_sim.field import SolenoidRegion
from spinecho_sim.solver import FieldSolver
from spinecho_sim.state import (
    CoherentSpin,
    MonatomicParticleState,
    get_expectation_values,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class SweepResult:
    """Results from a parameter sweep."""

    parameter_values: np.ndarray  # Values of the swept parameter
    results: np.ndarray[
        tuple[Literal[4], ...]
    ]  # Array of shape (4, num_params) for <Sx>, <Sy>, <Sz>, I_perp


def create_initial_states(  # noqa: PLR0913
    *,
    spin: float = 0.5,
    num_spins: int,
    velocity: float,
    velocity_spread: float,
    initial_theta: float = np.pi / 2,
    initial_phi: float = 0,
    gyromagnetic_ratio: float = -2.04e8,  # He-3 gyromagnetic ratio
    beam_radius: float = 1.16e-3,
) -> list[MonatomicParticleState]:
    """Create a list of initial particle states with given parameters."""
    assert np.isclose(2 * spin, int(2 * spin)), "Spin must be integer or half-integer."
    return [
        MonatomicParticleState(
            _spin_angular_momentum=CoherentSpin(
                theta=initial_theta, phi=initial_phi
            ).as_generic(n_stars=int(2 * spin)),
            displacement=displacement,
            parallel_velocity=v,
            gyromagnetic_ratio=gyromagnetic_ratio,
        )
        for displacement, v in zip(
            sample_uniform_displacement(num_spins, beam_radius),
            sample_gaussian_velocities(num_spins, velocity, velocity_spread),
            strict=True,
        )
    ]


def sweep_field_current(
    currents: np.ndarray,
    initial_states: list[MonatomicParticleState],
    solenoid_length: float = 0.75,
    magnetic_constant: float = 3.96e-3,
    n_steps: int = 1000,
) -> SweepResult:
    """Sweep over different magnetic field currents and record final spin states."""
    # Initialize arrays to store results
    results = np.asarray(np.zeros((4, len(currents))))

    weights = currents / currents[0]
    # Create field with the current parameter value
    field = SolenoidRegion.from_experimental_parameters(
        length=solenoid_length,
        magnetic_constant=magnetic_constant,
        current=currents[0],
    ).then(
        SolenoidRegion.from_experimental_parameters(
            length=solenoid_length,
            magnetic_constant=magnetic_constant,
            current=-currents[0],
        ).translate(dz=solenoid_length)
    )

    # Run simulations for each parameter value
    for i, weight in enumerate(tqdm(weights, desc="Sweeping field current")):
        # Run simulation
        weighted_field = field.scale(weight)
        solver = FieldSolver(region=weighted_field)
        result = solver.simulate_monatomic_trajectories(initial_states, n_steps=n_steps)

        expectation_values = get_expectation_values(result.spin)
        # Extract final spin expectation values
        sx, sy, sz = np.mean(expectation_values, axis=1)[:, -1]
        transverse_intensity = sx**2 + sy**2

        # Store results in the multidimensional array
        results[:, i] = np.asarray([sx, sy, sz, transverse_intensity])

    return SweepResult(
        parameter_values=currents,
        results=results,
    )


def plot_sweep_results(
    sweep_result: SweepResult,
    parameter_name: str = "Current (A)",
    save_path: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the results of a parameter sweep."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sweep_result.parameter_values,
        sweep_result.results[3],
        "k--",
        label=r"$|\langle \mathbf{S} \rangle|_\perp$",
    )

    ax.set_xlabel(parameter_name)
    ax.set_ylabel("Spin Expectation Value")
    ax.set_title("Final Spin States vs. Magnetic Field Strength")
    ax.legend()
    ax.grid(visible=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
