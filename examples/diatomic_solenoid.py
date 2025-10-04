from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.field import SolenoidRegion
from spinecho_sim.solver import (
    FieldSolver,
    # animate_diatomic_mean_expectation_vectors,
    plot_diatomic_alignment_diagnostics,
    # plot_diatomic_alignment_tensor,
    # plot_diatomic_expectation_differences,
    # plot_diatomic_expectation_values,
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
    num_spins = 1
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

    field = SolenoidRegion.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    solver = FieldSolver(region=field)
    result = solver.simulate_diatomic_trajectories(initial_states, n_steps=1000)

    hilbert_space_dimensions = result.hilbert_space_dims
    quantum_number_i = (hilbert_space_dimensions[0] - 1) / 2
    quantum_number_j = (hilbert_space_dimensions[1] - 1) / 2
    i_label = (
        f"{quantum_number_i:.0f}"
        if quantum_number_i is int
        else f"{quantum_number_i:.1f}"
    )
    j_label = (
        f"{quantum_number_j:.0f}"
        if quantum_number_j is int
        else f"{quantum_number_j:.1f}"
    )

    # fig, ax = plot_diatomic_expectation_values(result)
    # fig.suptitle(
    #     r"Nuclear Spin Expectation Values for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
    #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    #     f"{num_spins} spins, $I={i_label}$, $J={j_label}$",
    # )
    # output_path = f"./examples/diatomic_solenoid.expectation.{num_spins}-spins_I-{i_label}_J-{j_label}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    # fig, ax = plot_diatomic_expectation_differences(result)
    # fig.suptitle(
    #     r"Nuclear Spin Expectation Values for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
    #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    #     f"{num_spins} spins, $I={i_label}$, $J={j_label}$",
    # )
    # output_path = f"./examples/diatomic_solenoid.expectation_differences.{num_spins}-spins_I-{i_label}_J-{j_label}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    # anim = animate_diatomic_mean_expectation_vectors(result)
    # output_path = f"./examples/diatomic_solenoid.expectation.animations.{num_spins}-spins_I-{i_label}_J-{j_label}.mp4"

    # # Save the animation
    # anim.save(output_path, fps=60, writer="ffmpeg")  # Save as MP4 using ffmpeg
    # print(f"Animation saved to {output_path}")

    # fig, ax = plot_diatomic_alignment_tensor(result, "I")
    # fig.suptitle(
    #     r"Nuclear Spin Alignment Tensor $Q_{ij}$ for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
    #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    #     f"{num_spins} spins, $I={i_label}$, $J={j_label}$",
    # )
    # output_path = f"./examples/diatomic_solenoid.alignment.{num_spins}-spins_I-{i_label}_J-{j_label}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    fig, ax = plot_diatomic_alignment_diagnostics(result, "I")
    fig.suptitle(
        r"Measures of Nuclear Spin Alignment Tensor $Q_{ij}$ for H$_2$ Molecular Beam in a Sinusoidal Magnetic Field, "
        r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
        f"{num_spins} spins, $I={i_label}$, $J={j_label}$",
    )
    # output_path = f"./examples/diatomic_solenoid.q_tensor_measure.{num_spins}-spins_I-{i_label}_J-{j_label}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.show()
