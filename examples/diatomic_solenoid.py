from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.solenoid import (
    Solenoid,
    # animate_diatomic_mean_expectation_vectors,
    # plot_diatomic_expectation_differences,
    # plot_diatomic_expectation_values,
    plot_diatomic_alignment_tensor,
)
from spinecho_sim.state import (
    CoherentSpin,
    ParticleState,
    StateVectorParticleState,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 20
    initial_states = [
        StateVectorParticleState.from_spin_state(
            ParticleState(
                _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
                    n_stars=2
                ),
                _rotational_angular_momentum=CoherentSpin(
                    theta=np.pi / 2, phi=0
                ).as_generic(n_stars=2),
                displacement=displacement,
                parallel_velocity=velocity,
                coefficients=(
                    2 * np.pi * 4.258e7,
                    2 * np.pi * 0.66717e7,
                    2 * np.pi * 113.8e3,
                    2 * np.pi * 57.68e3,
                ),
            )
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
    result = solenoid.simulate_diatomic_trajectories(initial_states, n_steps=1000)

    hilbert_space_dimensions = result.hilbert_space_dims
    I = (hilbert_space_dimensions[0] - 1) / 2
    J = (hilbert_space_dimensions[1] - 1) / 2
    I_label = f"{I:.0f}" if I is int else f"{I:.1f}"
    J_label = f"{J:.0f}" if J is int else f"{J:.1f}"

    # fig, ax = plot_diatomic_expectation_values(result)
    # fig.suptitle(
    #     r"Nuclear Spin Expectation Values for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
    #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    #     f"{num_spins} spins, $I={I_label}$, $J={J_label}$",
    # )
    # output_path = f"./examples/classical_solenoid.expectation.{num_spins}-spins_I-{I_label}_J-{J_label}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    # # fig, ax = plot_diatomic_expectation_differences(result)
    # # fig.suptitle(
    # #     r"Nuclear Spin Expectation Values for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
    # #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    # #     f"{num_spins} spins, $I={I_label}$, $J={J_label}$",
    # # )
    # # output_path = f"./examples/classical_solenoid.expectation_differences.{num_spins}-spins_I-{I_label}_J-{J_label}.pdf"
    # # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    # anim = animate_diatomic_mean_expectation_vectors(result)
    # output_path = f"./examples/classical_solenoid.expectation.animations.{num_spins}-spins_I-{I_label}_J-{J_label}.mp4"

    # # Save the animation
    # anim.save(output_path, fps=60, writer="ffmpeg")  # Save as MP4 using ffmpeg
    # print(f"Animation saved to {output_path}")

    fig, ax = plot_diatomic_alignment_tensor(result, "I")
    fig.suptitle(
        r"Nuclear Spin Alignment Tensor $Q_{ij}$ for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $I={I_label}$, $J={J_label}$",
    )
    output_path = f"./examples/classical_solenoid.alignment.{num_spins}-spins_I-{I_label}_J-{J_label}.pdf"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()
