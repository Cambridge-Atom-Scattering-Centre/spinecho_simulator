from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import SolenoidSimulator
from spinecho_sim.classical_solenoid_test import (
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
    num_spins = 20
    dt = 1 / (100 * 2.04e8 * B_0)  # 100 time steps per Larmor period
    sim_params = {
        "velocity": sample_gaussian_velocities(
            num_spins, particle_velocity, 0.225 * particle_velocity
        ),
        # "field": [-0.01, 0.01, 1.0],
        "field": lambda z: [
            0.0,
            0.0,
            B_0 * np.sin(np.pi * np.asarray(z) / solenoid_length) ** 2,
        ],
        "time_step": dt,  # Time step in seconds
        "length": solenoid_length,
        "init_spin": sample_s_uniform(num_spins, np.array([0.5, 0.0, 0.0])),
        "perp_dist": sample_disk(num_spins, 1.16e-3),  # Sample a disk of radius 0.1
    }
    sim = SolenoidSimulator(sim_params)
    z, s = sim.run()

    fig, ax = plt.subplots(figsize=(10, 6))

    # s is assumed to be (num_timesteps, num_spins, 3)
    num_spins = s.shape[0]

    # Assign a distinct colormap for each spin component

    colormaps = [
        plt.cm.Blues,  # For Sx
        plt.cm.Greens,  # For Sy
        plt.cm.Reds,  # For Sz
    ]
    # For each component, generate a set of shades for all spins
    component_colors = [cmap(np.linspace(0.3, 0.8, num_spins)) for cmap in colormaps]

    # Choose a common z-grid
    Nz = 500  # number of distance bins you want
    z_grid = np.linspace(0, sim_params.get("length"), Nz)

    # Interpolate each spin’s angles onto that z-grid, You'll end up with shape (Nz, Nspins, 3)
    S_on_z = np.empty((Nz, num_spins, 3))
    Angles_on_z = np.empty((Nz, num_spins, 2))

    for spin_idx in range(num_spins):
        # Get the number of time steps for this spin
        n_steps = len(z[spin_idx])
        # Plot only the valid segment for this spin
        theta = np.arctan2(s[spin_idx, :n_steps, 1], s[spin_idx, :n_steps, 0])
        phi = np.arctan2(
            s[spin_idx, :n_steps, 2],
            np.sqrt(s[spin_idx, :n_steps, 0] ** 2 + s[spin_idx, :n_steps, 1] ** 2),
        )
        ax.plot(
            z[spin_idx] - solenoid_length / 2,
            theta,
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[0][spin_idx],
        )
        ax.plot(
            z[spin_idx] - solenoid_length / 2,
            phi,
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[1][spin_idx],
        )

        # for each component separately:
        for comp in range(3):
            S_on_z[:, spin_idx, comp] = np.interp(
                z_grid,  # the z-coordinates where to interpolate
                z[spin_idx],  # each spin’s sampled distances
                s[spin_idx, :n_steps, comp],  # the values to interpolate
            )

        Angles_on_z[:, spin_idx, 0] = np.interp(
            z_grid,  # the z-coordinates where to interpolate
            z[spin_idx],  # each spin’s sampled distances
            theta,  # the values to interpolate
        )
        Angles_on_z[:, spin_idx, 1] = np.interp(
            z_grid,  # the z-coordinates where to interpolate
            z[spin_idx],  # each spin’s sampled distances
            phi,  # the values to interpolate
        )

    Angles_avg = Angles_on_z.mean(axis=1)  # shape (Nz, 3)
    Angles_std = Angles_on_z.std(axis=1)  # if you want error-bars

    ax.plot(
        z_grid - solenoid_length / 2,
        Angles_avg[:, 0],
        label=r"⟨$\theta$⟩",
        color=component_colors[0][spin_idx],
    )
    ax.plot(
        z_grid - solenoid_length / 2,
        Angles_avg[:, 1],
        label=r"⟨$\phi$⟩",
        color=component_colors[1][spin_idx],
    )

    # Plot fill_between for Sx, Sy, Sz above all previous plots by setting higher zorder
    ax.fill_between(
        z_grid - solenoid_length / 2,
        Angles_avg[:, 0] - Angles_std[:, 0],
        Angles_avg[:, 0] + Angles_std[:, 0],
        alpha=0.2,
        zorder=10,
        label=r"⟨$\theta$⟩ ± std",
        color=component_colors[0][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )
    ax.fill_between(
        z_grid - solenoid_length / 2,
        Angles_avg[:, 1] - Angles_std[:, 1],
        Angles_avg[:, 1] + Angles_std[:, 1],
        alpha=0.2,
        zorder=10,
        label=r"⟨$\phi$⟩ ± std",
        color=component_colors[1][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )

    ax.axhline(np.pi, color="black", linestyle="--", linewidth=0.5, label=r"$\pi$")
    ax.axhline(-np.pi, color="black", linestyle="--", linewidth=0.5, label=r"$-\pi$")

    ax.set_xlim(-solenoid_length / 2, solenoid_length / 2)
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Phase Angles (radians)")
    ax.set_title(
        r"Classical Larmor Precession in a Sinusoidal Magnetic Field $\mathbf{B} \approx B_0 \mathbf{z}$,"
        f" {num_spins} spins,"
        f" {int(solenoid_length / (dt * particle_velocity))} steps"
    )
    ax.legend()
    ax.grid(visible=True)

    # Save the plot instead of showing it
    output_path = "/workspaces/spinecho_sim/figures/classical_solenoid_angles_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
