from __future__ import annotations

import matplotlib.pyplot as plt

from spinecho_sim import SolenoidSimulator

if __name__ == "__main__":
    print("Example placeholder script for spinecho simulation.")
    sim_params = {
        "velocity": 1.0,
        "field": [0.0, 0.0, 1.0],
        "time_step": 0.01,
        "length": 10.0,
        "init_spin": [1.0, 0.0, 0.0],
    }
    sim = SolenoidSimulator(sim_params)
    t, s = sim.run()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, s[:, 0], label=r"$m_x$")
    ax.plot(t, s[:, 1], label=r"$m_y$")
    ax.plot(t, s[:, 2], label=r"$m_z$")

    ax.set_xlabel("Time")
    ax.set_ylabel("Spin components")
    ax.set_title(
        r"Classical Larmor Precession in a Uniform Magnetic Field $\mathbf{B}=B_0 \mathbf{z}$"
    )
    ax.legend()
    ax.grid(visible=True)

    plt.show()
