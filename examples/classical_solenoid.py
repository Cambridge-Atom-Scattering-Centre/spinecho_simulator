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
    B_0 = magnetic_constant * current / (0.636 * solenoid_length)
    num_spins = 20
    dt = 1 / (100 * 2.04e8 * B_0)  # 100 time steps per Larmor period

    Solenoid = Solenoid(
        solenoid_length,
        dt,
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

    # Choose a common z-grid
    Nz = 500  # number of distance bins you want
    z_grid = np.linspace(0, Solenoid.length, Nz)

    # Interpolate each spin’s components onto that z-grid, You'll end up with shape (Nz, Nspins, 3)
    S_on_z = np.empty((Nz, num_spins, 3))

    for spin_idx in range(num_spins):
        # Get the number of time steps for this spin
        n_steps = len(z[spin_idx])
        # Plot only the valid segment for this spin
        ax.plot(
            z[spin_idx] - solenoid_length / 2,
            s[spin_idx, :n_steps, 0],
            # label=rf"$m_x$ (spin {spin_idx + 1})",
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[0][spin_idx],
        )
        ax.plot(
            z[spin_idx] - solenoid_length / 2,
            s[spin_idx, :n_steps, 1],
            # label=rf"$m_y$ (spin {spin_idx + 1})",
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[1][spin_idx],
        )
        ax.plot(
            z[spin_idx] - solenoid_length / 2,
            s[spin_idx, :n_steps, 2],
            # label=rf"$m_z$ (spin {spin_idx + 1})",
            linewidth=1.0,
            alpha=0.2,
            color=component_colors[2][spin_idx],
        )

        # for each component separately:
        for comp in range(3):
            S_on_z[:, spin_idx, comp] = np.interp(
                z_grid,  # the z-coordinates where to interpolate
                z[spin_idx],  # each spin’s sampled distances
                s[spin_idx, :n_steps, comp],  # the values to interpolate
            )

    S_avg = S_on_z.mean(axis=1)  # shape (Nz, 3)
    S_std = S_on_z.std(axis=1)  # if you want error-bars

    ax.plot(
        z_grid - Solenoid.length / 2,
        S_avg[:, 0],
        label="⟨Sx⟩",
        color=component_colors[0][spin_idx],
    )
    ax.plot(
        z_grid - Solenoid.length / 2,
        S_avg[:, 1],
        label="⟨Sy⟩",
        color=component_colors[1][spin_idx],
    )
    ax.plot(
        z_grid - Solenoid.length / 2,
        S_avg[:, 2],
        label="⟨Sz⟩",
        color=component_colors[2][spin_idx],
    )
    # Plot fill_between for Sx, Sy, Sz above all previous plots by setting higher zorder
    ax.fill_between(
        z_grid - Solenoid.length / 2,
        S_avg[:, 0] - S_std[:, 0],
        S_avg[:, 0] + S_std[:, 0],
        alpha=0.2,
        zorder=10,
        label="⟨Sx⟩ ± std",
        color=component_colors[0][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )
    ax.fill_between(
        z_grid - Solenoid.length / 2,
        S_avg[:, 1] - S_std[:, 1],
        S_avg[:, 1] + S_std[:, 1],
        alpha=0.2,
        zorder=10,
        label="⟨Sy⟩ ± std",
        color=component_colors[1][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )
    ax.fill_between(
        z_grid - Solenoid.length / 2,
        S_avg[:, 2] - S_std[:, 2],
        S_avg[:, 2] + S_std[:, 2],
        alpha=0.2,
        zorder=10,
        label="⟨Sz⟩ ± std",
        color=component_colors[2][spin_idx],
        linewidth=0.5,
        linestyle="--",
    )

    Intensity = (S_avg[:, 0] ** 2 + S_avg[:, 1] ** 2) / (0.5**2)
    ax.plot(
        z_grid - Solenoid.length / 2,
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
        f" {num_spins} spins,"
        f" {int(Solenoid.length / (Solenoid.time_step * particle_velocity))} steps"
    )
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(visible=True)

    # Save the plot instead of showing it
    output_path = "./examples/classical_solenoid.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
