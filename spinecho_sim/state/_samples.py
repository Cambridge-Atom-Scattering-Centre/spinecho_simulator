from __future__ import annotations

from typing import Any

import numpy as np

from spinecho_sim.state._displacement import (
    ParticleDisplacement,
    ParticleDisplacementList,
)
from spinecho_sim.state._spin import (
    CoherentSpin,
    Spin,
)


def sample_uniform_spin(n: int) -> Spin[tuple[int]]:
    """Sample N uniform random directions on the unit sphere."""
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=n)
    theta = rng.uniform(0, np.pi, size=n)
    return Spin.from_iter(
        CoherentSpin(theta=t, phi=p) for t, p in zip(theta, phi, strict=True)
    )


def sample_s_unit_circle(n: int) -> Spin[tuple[int]]:
    """Sample N uniform random directions on the unit circle normal to z-axis."""
    rng = np.random.default_rng()
    phi = rng.uniform(0, 2 * np.pi, size=n)
    return Spin.from_iter(CoherentSpin(theta=np.pi / 2, phi=p) for p in phi)


def sample_uniform_displacement(n: int, r_max: float) -> ParticleDisplacementList:
    """Sample N random radii uniformly distributed over a disk of radius r_max."""
    rng = np.random.default_rng()
    # angles uniform on [0,2π)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    # radii with PDF ∝ r  →  CDF ∝ r^2  →  r = R * sqrt(u)
    u = rng.uniform(0, 1, size=n)
    r = r_max * np.sqrt(u)
    return ParticleDisplacementList.from_displacements(
        [
            ParticleDisplacement(r=r_i, theta=theta_i)
            for r_i, theta_i in zip(r, theta, strict=True)
        ]
    )


def sample_boltzmann_velocities(
    n: int, temperature: float, mass: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Sample N velocities from a Boltzmann distribution."""
    k_b = 1  # 1.380649e-23  # Boltzmann constant (J/K)

    sigma = np.sqrt(k_b * temperature / mass)
    rng = np.random.default_rng()
    velocities = rng.normal(loc=0, scale=sigma, size=(n))
    return np.linalg.norm(velocities, axis=1)


def sample_gaussian_velocities(
    n: int, average_velocity: float, std_velocity: float
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Sample N velocities from a Gaussian distribution."""
    rng = np.random.default_rng()
    return rng.normal(loc=average_velocity, scale=std_velocity, size=n)
