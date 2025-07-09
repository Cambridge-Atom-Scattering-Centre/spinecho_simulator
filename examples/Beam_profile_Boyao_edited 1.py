from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import norm

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Constants
gamma_he = 2.037895e8  # gyromagnetic ratio of 3He (s^-1 T^-1)
hbar = 1.05457e-34  # reduced Planck constant (J s)

# Simulation parameters
current_range = np.arange(0.7, 2.3001, 0.004)
r_values = np.array([0.001])
theta_values = np.array([90.0])
dz = 0.01
z = np.arange(-1.5, 1.5 + dz, dz)
dv = 2
velocities = np.arange(600, 800 + dv, dv)


def magnetic_field(I1, r, z, theta_deg):
    mu0 = 1.2566e-6
    h1 = 0.72
    h2 = h1 * 995 / 1370
    r1 = 0.025
    r2 = 0.045

    B01 = 1370 / h1 * I1 * mu0
    B02 = 1370 / h1 * 2 * mu0

    Bz1 = (B01 / 2) * (
        ((z + h1 / 2) / np.sqrt(r1**2 + (z + h1 / 2) ** 2))
        - ((z - h1 / 2) / np.sqrt(r1**2 + (z - h1 / 2) ** 2))
    )
    Bz2 = (B02 / 2) * (
        ((z + h2 / 2) / np.sqrt(r2**2 + (z + h2 / 2) ** 2))
        - ((z - h2 / 2) / np.sqrt(r2**2 + (z - h2 / 2) ** 2))
    )
    Bz = Bz1 - Bz2

    Br1 = (B01 / 4) * (
        (r * r1**2) / ((r1**2 + (z - h1 / 2) ** 2) ** 1.5)
        - (r * r1**2) / ((r1**2 + (z + h1 / 2) ** 2) ** 1.5)
    )
    Br2 = (B02 / 4) * (
        (r * r2**2) / ((r2**2 + (z - h2 / 2) ** 2) ** 1.5)
        - (r * r2**2) / ((r2**2 + (z + h2 / 2) ** 2) ** 1.5)
    )
    Br = Br1 - Br2

    theta = np.deg2rad(theta_deg)
    Bx = Br * np.cos(theta)
    By = Br * np.sin(theta)

    return Bx, By, Bz


# Pre-allocate result
beam_profile = np.empty_like(current_range)

# Main loop
for idx, I1 in enumerate(current_range):
    sum_total = 0.0
    for r in r_values:
        for theta in theta_values:
            P_sum = 0.0
            Bx, By, Bz = magnetic_field(I1, r, z, theta)
            for v in velocities:
                # initial spin state (column vector)
                state = np.array([np.sqrt(0.5), 1j * np.sqrt(0.5)], dtype=complex)
                for Bx_k, By_k, Bz_k in zip(Bx, By, Bz, strict=False):
                    H = (
                        -gamma_he
                        * hbar
                        / 2
                        * (Bx_k * sigma_x + By_k * sigma_y + Bz_k * sigma_z)
                    )
                    U = expm(-1j * H * dz / (v * hbar))
                    state = U @ state
                # compute expectation value of (Py - Pz)/sqrt(2)
                P_op = (sigma_y - sigma_z) / np.sqrt(2)
                Py = state.conj().T @ P_op @ state
                P_sum += 2 * dv * Py * norm.pdf(v, 715, 0.03 * 715)
            sum_total += r * P_sum
    # Normalize
    beam_profile[idx] = np.real(sum_total) / np.mean(r_values) / len(theta_values)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(current_range, beam_profile)
plt.xlabel("I_inner (A)")
plt.ylabel("Spin polarisation")
plt.title("Beam Profile vs Inner Current")
plt.grid(True)

# Save the plot instead of showing it
output_path = "/workspaces/spinecho_sim/figures/Beam_profile_Boyao_1.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
