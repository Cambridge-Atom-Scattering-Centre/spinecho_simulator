from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim.classical_solenoid_test import (
    Solenoid,
    sample_disk,
    sample_gaussian_velocities,
    sample_s_uniform,
)

if __name__ == "__main__":
    print("Script for Classical Solenoid simulation of spin angles.")
    particle_velocity = 714  # m/s
    magnetic_constant = 3.96e-3
    solenoid_length = 0.75  # Length of the solenoid in meters
    current = 0.01  # Current in Amperes
    B_0 = np.pi * magnetic_constant * current / (2 * solenoid_length)
    num_spins = 50
    dx = particle_velocity / (100 * 2.04e8 * B_0)  # 100 time steps per Larmor period

    params = Solenoid(
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
    result = solenoid.simulate_trajectory()

    # Precompute shifted z
    z_shifted = z - solenoid_length / 2

    theta_wrapped = np.arctan2(s[:, :, 1], s[:, :, 0])
    phi_wrapped = np.arctan2(
        s[:, :, 2],
        np.sqrt(s[:, :, 0] ** 2 + s[:, :, 1] ** 2),
    )  # shape (num_spins, steps)

    # Unwrap angles along the time axis (axis=1)
    theta = np.unwrap(theta_wrapped, axis=1)
    phi = np.unwrap(phi_wrapped, axis=1)

    # Assign to Angles_on_z (shape: steps, num_spins, 2)
    angles = np.stack([theta.T, phi.T], axis=2)  # shape (steps, num_spins, 2)

    # Compute mean and std over spins for each step and component
    Angles_avg = angles.mean(axis=1)  # shape (steps, 2)
    Angles_std = angles.std(axis=1)  # shape (steps, 2)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot averages and std bands
    for comp, label, color in zip(
        range(2), [r"$\theta$", r"$\phi$"], ["blue", "red"], strict=False
    ):
        ax.plot(
            z_shifted,
            (angles[:, :, comp] / np.pi),
            linewidth=1.0,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            z_shifted,
            Angles_avg[:, comp] / np.pi,
            label=label,
            color=color,
        )
        ax.fill_between(
            z_shifted,
            (Angles_avg[:, comp] - Angles_std[:, comp]) / np.pi,
            (Angles_avg[:, comp] + Angles_std[:, comp]) / np.pi,
            alpha=0.2,
            zorder=10,
            label=label + r" ± std",
            color=color,
            linewidth=0.5,
            linestyle="--",
        )

    ax.set_xlim(-Solenoid.length / 2, Solenoid.length / 2)
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel(r"Phase Angles (radians/$\pi$)")
    ax.set_title(
        r"Classical Larmor Precession in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$,"
        f" {num_spins} spins,"
        f" {int(Solenoid.length / dx)} steps"
    )
    ax.legend()
    ax.grid(visible=True)

    # Save the plot instead of showing it
    output_path = "./examples/classical_solenoid_angle.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
