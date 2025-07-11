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
    print("Script for Classical Solenoid simulation of spin angles.")
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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Assign a distinct colormap for each spin component

    colormaps = [
        plt.cm.Blues,  # For Sx
        plt.cm.Greens,  # For Sy
        plt.cm.Reds,  # For Sz
    ]
    # For each component, generate a set of shades for all spins
    component_colors = [cmap(np.linspace(0.3, 0.8, num_spins)) for cmap in colormaps]

    Angles_on_z = np.zeros((len(z), num_spins, 2))

    for spin_idx in range(num_spins):
        # Get the number of time steps for this spin
        n_steps = len(z)
        # Plot only the valid segment for this spin
        theta_wrapped = np.arctan2(s[spin_idx, :, 1], s[spin_idx, :, 0])
        phi_wrapped = np.arctan2(
            s[spin_idx, :, 2],
            np.sqrt(s[spin_idx, :, 0] ** 2 + s[spin_idx, :, 1] ** 2),
        )

        theta = np.unwrap(theta_wrapped, np.pi)
        phi = np.unwrap(phi_wrapped, np.pi)

        ax.plot(
            z - solenoid_length / 2,
            theta / np.pi,
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[0][spin_idx],
        )
        ax.plot(
            z - solenoid_length / 2,
            phi / np.pi,
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[1][spin_idx],
        )

        Angles_on_z[:, spin_idx, 0] = theta
        Angles_on_z[:, spin_idx, 1] = phi

    Angles_avg = Angles_on_z.mean(axis=1)  # shape (Nz, 3)
    Angles_std = Angles_on_z.std(axis=1)  # if you want error-bars

    ax.plot(
        z - Solenoid.length / 2,
        Angles_avg[:, 0] / np.pi,
        label=r"⟨$\theta$⟩",
        color=component_colors[0][spin_idx],
    )
    ax.plot(
        z - Solenoid.length / 2,
        Angles_avg[:, 1] / np.pi,
        label=r"⟨$\phi$⟩",
        color=component_colors[1][spin_idx],
    )

    # Plot fill_between for Sx, Sy, Sz above all previous plots by setting higher zorder
    ax.fill_between(
        z - Solenoid.length / 2,
        (Angles_avg[:, 0] - Angles_std[:, 0]) / np.pi,
        (Angles_avg[:, 0] + Angles_std[:, 0]) / np.pi,
        alpha=0.2,
        zorder=10,
        label=r"⟨$\theta$⟩ ± std",
        color=component_colors[0][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )
    ax.fill_between(
        z - Solenoid.length / 2,
        (Angles_avg[:, 1] - Angles_std[:, 1]) / np.pi,
        (Angles_avg[:, 1] + Angles_std[:, 1]) / np.pi,
        alpha=0.2,
        zorder=10,
        label=r"⟨$\phi$⟩ ± std",
        color=component_colors[1][spin_idx],
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
