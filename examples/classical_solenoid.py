from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import SolenoidSimulator
from spinecho_sim.classical_solenoid_test import (
    Solenoid,
    sample_disk,
    sample_gaussian_velocities,
    sample_s_uniform,
)

if __name__ == "__main__":
    print("Script for Classical Solenoid simulation of spin components.")
    particle_velocity = 714  # m/s
    magnetic_constant = 3.96e-3
    solenoid_length = 0.75  # Length of the solenoid in meters
    current = 0.01  # Current in Amperes
    B_0 = np.pi * magnetic_constant * current / (2 * solenoid_length)
    num_spins = 50
    dx = particle_velocity / (100 * 2.04e8 * B_0)  # 100 time steps per Larmor period

    Solenoid = Solenoid(
        solenoid_length,
        dx,
        sample_gaussian_velocities(
            num_spins, particle_velocity, 0.225 * particle_velocity
        ),
        sample_s_uniform(num_spins, np.array([0.5, 0.0, 0.0])),
        lambda z: np.array(
            [
                0.0,
                0.0,
                B_0 * np.sin(np.pi * np.asarray(z) / solenoid_length) ** 2,
            ]
        ),
        sample_disk(num_spins, 1.16e-3),
    )
    sim = SolenoidSimulator(Solenoid)
    z, s = sim.run()

    S_avg = s.mean(axis=0)
    S_std = s.std(axis=0)

    # Precompute shifted z
    z_shifted = z - Solenoid.length / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot averages and std bands
    for comp, label, color in zip(
        range(3), ["⟨Sx⟩", "⟨Sy⟩", "⟨Sz⟩"], ["blue", "green", "red"], strict=False
    ):
        ax.plot(
            z_shifted,
            s[:, :, comp].T,  # shape (num_spins, steps)
            linewidth=1.0,
            alpha=0.1,
            color=color,  # use a mid-tone for all
        )
        ax.plot(
            z_shifted,
            S_avg[:, comp],
            label=label,
            color=color,
        )
        ax.fill_between(
            z_shifted,
            S_avg[:, comp] - S_std[:, comp],
            S_avg[:, comp] + S_std[:, comp],
            alpha=0.2,
            zorder=10,
            label=f"{label} ± std",
            color=color,
            linewidth=0.5,
            linestyle="--",
        )

    Intensity = (S_avg[:, 0] ** 2 + S_avg[:, 1] ** 2) / (0.5**2)
    ax.plot(
        z_shifted,
        Intensity,
        label=r"$I_\parallel =\langle S_x\rangle^2+\langle S_y\rangle^2$",
        color="black",
        linestyle="--",
    )

    ax.set_xlim(-Solenoid.length / 2, Solenoid.length / 2)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin components")
    ax.set_title(
        r"Classical Larmor Precession in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$,"
        f" {current}A, {num_spins} spins, {int(Solenoid.length / dx)} steps"
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(visible=True)

    # Save the plot instead of showing it
    output_path = "./examples/classical_solenoid.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
