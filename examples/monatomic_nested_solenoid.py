from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import create_initial_states
from spinecho_sim.field import FieldRegion, SolenoidRegion
from spinecho_sim.solver import (
    FieldSolver,
    plot_monatomic_expectation_angles,
    plot_monatomic_expectation_values,
    plot_monatomic_spin_states,
)


def create_nested_solenoids(
    *,
    outer_length: float,
    inner_length: float,
    magnetic_constant: float,
    outer_current: float,
    inner_current: float,
) -> FieldRegion:
    """Create a field with one solenoid and another shorter one inside with reversed current."""
    # Create the outer solenoid
    outer_solenoid = SolenoidRegion.from_experimental_parameters(
        length=outer_length,
        magnetic_constant=magnetic_constant,
        current=outer_current,
    )

    # Create the inner solenoid, translated to be centered inside the outer solenoid
    inner_solenoid = SolenoidRegion.from_experimental_parameters(
        length=inner_length,
        magnetic_constant=magnetic_constant,
        current=inner_current,
        z_start=outer_length / 2 - inner_length / 2,
    )

    # Combine the two solenoids into a single field
    return outer_solenoid + inner_solenoid


if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 10
    initial_states = create_initial_states(
        spin=0.5,
        num_spins=num_spins,
        velocity=particle_velocity,
        velocity_spread=0.225 * particle_velocity,
        initial_theta=np.pi / 2,
        initial_phi=0,
        gyromagnetic_ratio=-2.04e8,
        beam_radius=1.16e-3,
    )

    field = create_nested_solenoids(
        outer_length=0.75,
        inner_length=0.375,
        magnetic_constant=3.96e-3,
        outer_current=0.1,
        inner_current=-0.1,
    )
    solver = FieldSolver(region=field)
    result = solver.simulate_monatomic_trajectories(initial_states, n_steps=1000)

    n_stars = result.spin.n_stars
    S = n_stars / 2
    S_label = f"{S:.0f}" if S is int else f"{int(2 * S)}/2"

    fig, ax = plot_monatomic_spin_states(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/monatomic_nested_solenoid.state.{num_spins}-spins_S-{S}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_monatomic_expectation_values(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/monatomic_nested_solenoid.expectation.{num_spins}-spins_S-{S}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_monatomic_expectation_angles(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $S={S_label}$",
    )
    output_path = (
        f"./examples/monatomic_nested_solenoid.angles.{num_spins}-spins_S-{S}.pdf"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()
