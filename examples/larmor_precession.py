from __future__ import annotations

# import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.solenoid import (
    MonatomicSolenoid,
    animate_monatomic_mean_expectation_vectors,
)
from spinecho_sim.state import (
    CoherentSpin,
    MonatomicParticleState,
    ParticleDisplacement,
)

if __name__ == "__main__":
    # When a particle is placed in a solenoid, it will precess around the magnetic field axis.
    # if it is displaced from the center of the solenoid, it will also slightly
    # rotate towards s_z, which reduces the intensity of the beam.
    particle_velocity = 714
    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2.5, phi=0).as_generic(
            n_stars=1
        ),
        parallel_velocity=714,
        displacement=ParticleDisplacement(r=1.16e-3),
    )

    solenoid = MonatomicSolenoid.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.1,
    )
    result = solenoid.simulate_trajectories([initial_state], n_steps=1000)

    n_stars = result.spin.n_stars
    S = n_stars / 2
    S_label = f"{S:.0f}" if S is int else f"{S:.1f}"

    # fig, ax = plot_monatomic_expectation_trajectories(result)
    # fig.suptitle(
    #     r"Classical Larmor Precession of ${}^3$He in a Sinusoidal Magnetic Field, "
    #     r"$\mathbf{{B}} \approx B_0 \mathbf{z}$, "
    #     f"$S={S_label}$",
    # )
    # output_path = f"./examples/larmor_precession.trajectory.{n_stars}.pdf"
    # plt.savefig(output_path, dpi=600, bbox_inches="tight")

    anim = animate_monatomic_mean_expectation_vectors(result)
    output_path = f"./examples/classical_solenoid.expectation.animations.{n_stars}.mp4"

    # Save the animation
    anim.save(output_path, fps=60, writer="ffmpeg")  # Save as MP4 using ffmpeg
    print(f"Animation saved to {output_path}")

    # plt.show()
