from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    ParticleState,
    Solenoid,
)
from spinecho_sim.solenoid import (
    plot_expectation_trajectory_3d,
    plot_expectation_values,
    plot_spin_states,
)
from spinecho_sim.state import (
    CoherentSpin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 10
    initial_states = [
        ParticleState(
            spin=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(n_stars=2),
            # spin=Spin.from_momentum_state(np.array([1, -1])),
            displacement=displacement,
            parallel_velocity=velocity,
        )
        for displacement, velocity in zip(
            sample_uniform_displacement(num_spins, 1.16e-3),
            sample_gaussian_velocities(
                num_spins, particle_velocity, 0.225 * particle_velocity
            ),
            strict=True,
        )
    ]

    solenoid = Solenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    result = solenoid.simulate_trajectories(initial_states, n_steps=1000)

    fig, ax = plot_spin_states(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins",
    )
    output_path = "./examples/classical_solenoid.state.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_expectation_values(result)
    fig.suptitle(
        r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins",
    )
    output_path = "./examples/classical_solenoid.expectation.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fix, ax = plot_expectation_trajectory_3d(result)

    plt.show()
