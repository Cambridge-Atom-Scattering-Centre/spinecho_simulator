from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spinecho_sim import (
    create_initial_states,
    plot_sweep_results,
    sweep_field_current,
)

if __name__ == "__main__":
    particle_velocity = 714
    num_spins = 50
    initial_states = create_initial_states(
        spin=0.5,
        num_spins=num_spins,
        velocity=particle_velocity,
        velocity_spread=0.225 * particle_velocity,
        initial_theta=np.pi / 2,
        initial_phi=0,
        gyromagnetic_ratio=-2.04e8,
        beam_radius=1.16e-3,
    )
    # Define the parameter range for the sweep
    currents = np.linspace(0.01, 1, 10)  # Current values

    # Run the sweep
    start_time = time.time()
    results = sweep_field_current(
        initial_states=initial_states,
        currents=currents,
        n_steps=500,  # Reduced for faster execution
    )
    elapsed = time.time() - start_time
    print(f"Sweep completed in {elapsed:.2f} seconds")

    # Create output directory if it doesn't exist
    Path("./examples/sweep_results").mkdir(parents=True, exist_ok=True)

    # Plot the results
    fig1, ax1 = plot_sweep_results(
        results,
        parameter_name="Current (A)",
        save_path="./examples/sweep_results/final_spin_vs_current.png",
    )
    plt.show()
