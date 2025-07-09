from __future__ import annotations

import csv

import numpy as np

from spinecho_sim import SolenoidSimulator
from spinecho_sim.classical_solenoid_test import (
    sample_disk,
    sample_gaussian_velocities,
    sample_s_uniform,
)

if __name__ == "__main__":
    print(
        "Script for Classical Solenoid simulation of depolarisation for varying field strengths."
    )
    # Parameter sweep
    B_fields = np.linspace(
        2, 10, 50
    )  # Magnetic field strengths from 0 to 2 (arbitrary units)
    final_intensities = []

    for B in B_fields:
        # Simulate the solenoid with the current magnetic field strength
        sim_params = {
            "velocity": sample_gaussian_velocities(50, 1.0, 0.008),
            "field": lambda z: [
                0.0,
                0.0,
                B * np.sin(np.pi * np.asarray(z) / 10) ** 2,
            ],
            "time_step": 0.01,
            "length": 10.0,
            "init_spin": sample_s_uniform(50, np.array([1.0, 0.0, 0.0])),
            "perp_dist": sample_disk(50, 0.05),  # Sample a disk of radius 0.05
        }
        sim = SolenoidSimulator(sim_params)
        z, s = sim.run()

        num_spins = s.shape[0]
        S_final = np.array([s[i, len(z[i]) - 1, :] for i in range(num_spins)])

        S_avg = S_final.mean(axis=0)
        S_final.std(axis=0)  # if you want error-bars

        # Calculate final intensity as the average of the squared components
        Intensity = S_avg[0] ** 2 + S_avg[1] ** 2

        # Run the simulation and store the final intensity
        final_intensities.append(Intensity)
        print(
            f"Simulated B-field {B:.2f}, Final Intensity: {final_intensities[-1]:.4f}"
        )
        del s, z, S_final

    # Plotting
    output_csv = (
        "/workspaces/spinecho_sim/data/classical_solenoid_field-vs-intensity_data2.csv"
    )
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["B_field", "Final_Intensity"])
        for B, intensity in zip(B_fields, final_intensities, strict=False):
            writer.writerow([B, intensity])
    print(f"Data saved to: {output_csv}")
