from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim import SolenoidSimulator
from spinecho_sim.classical_solenoid_test import sample_unit_circle

if __name__ == "__main__":
    print("Example placeholder script for spinecho simulation.")
    sim_params = {
        "velocity": 1.0,
        "field": [0.0, 0.0, 1.0],
        "time_step": 0.01,
        "length": 10.0,
        "init_spin": sample_unit_circle(2),  # Sample a random initial spin direction
    }
    sim = SolenoidSimulator(sim_params)
    t, s = sim.run()

    fig, ax = plt.subplots(figsize=(10, 6))

    # s is assumed to be (num_timesteps, num_spins, 3)
    num_spins = s.shape[1]
    for spin_idx in range(num_spins):
        ax.plot(t, s[:, spin_idx, 0], label=rf"$m_x$ (spin {spin_idx + 1})")
        ax.plot(t, s[:, spin_idx, 1], label=rf"$m_y$ (spin {spin_idx + 1})")
        ax.plot(t, s[:, spin_idx, 2], label=rf"$m_z$ (spin {spin_idx + 1})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Spin components")
    ax.set_title(
        r"Classical Larmor Precession in a Uniform Magnetic Field $\mathbf{B}=B_0 \mathbf{z}$"
    )
    ax.legend()
    ax.grid(visible=True)

    # Save the plot instead of showing it
    output_path = "/workspaces/spinecho_sim/figures/spin_simulation_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
