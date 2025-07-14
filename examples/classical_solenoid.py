from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    ParticleState,
    Solenoid,
)
from spinecho_sim.solenoid import plot_spin_components
from spinecho_sim.state import (
    CoherentSpin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 50
    initial_states = [
        ParticleState(
            spin=CoherentSpin(theta=np.pi / 2, phi=0),
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
        current=0.01,
    )
    result = solenoid.simulate_trajectories(initial_states)

    fig, ax = plot_spin_components(result)
    ax.set_title(
        r"Classical Larmor Precession in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$,"
        f"{num_spins} spins"
    )
    ax.legend(loc="lower right", fontsize="small")
    plt.show()

    output_path = "./examples/classical_solenoid.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
