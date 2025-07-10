"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

gyromagnetic_ratio = 2.04e8  # gyromagnetic ratio (rad s^-1 T^-1)


# -- helper: sample N uniform directions on the unit sphere --
def sample_s_unit_sphere(n: int) -> NDArray[np.floating[Any]]:
    """Sample N uniform random directions on the unit sphere.

    Returns
    -------
    NDArray[np.floating[Any]]
        An array of shape (N, 3) containing unit vectors.
    """
    phi: NDArray[np.floating[Any]] = np.random.uniform(0, 2 * np.pi, size=n)
    cos = np.random.uniform(-1, 1, size=n)
    sin = np.sqrt(1 - cos**2)
    return np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], axis=1)


# -- helper: sample N uniform directions on the unit sphere --
def sample_s_unit_circle(n: int) -> NDArray[np.floating[Any]]:
    """Sample N uniform random directions on the unit circle normal to z-axis.

    Returns
    -------
    NDArray[np.floating[Any]]
        An array of shape (N, 3) containing unit vectors.
    """
    phi: NDArray[np.floating[Any]] = np.random.uniform(0, 2 * np.pi, size=n)
    return np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)


def sample_disk(n: int, r_max: float) -> np.ndarray:
    """Sample N random radii uniformly distributed over a disk of radius r_max.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    r_max : float
        Maximum radius of the disk.

    Returns
    -------
    np.ndarray
        Array of sampled points with shape (N, 3): columns are x, y, theta.
    """
    # angles uniform on [0,2π)
    theta = np.random.uniform(0, 2 * np.pi, size=n)
    # radii with PDF ∝ r  →  CDF ∝ r^2  →  r = R * sqrt(u)
    u = np.random.uniform(0, 1, size=n)
    r = r_max * np.sqrt(u)
    return np.stack([r, theta], axis=1)


# -- helper: sample N uniform directions on the unit sphere --
def sample_s_uniform(
    n: int, spin: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Return an array of shape (n, 3) where each row is the given spin vector.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    spin : NDArray[np.floating[Any]]
        The spin vector to replicate.

    Returns
    -------
    NDArray[np.floating[Any]]
        An array of shape (n, 3) containing copies of the spin vector.
    """
    return np.tile(spin, (n, 1))


def sample_boltzmann_velocities(
    n: int, temp: float, mass: float
) -> NDArray[np.floating[Any]]:
    k_b = 1  # 1.380649e-23  # Boltzmann constant (J/K)

    def f(
        v: NDArray[np.floating[Any]], temp: float, mass: float
    ) -> NDArray[np.floating[Any]]:
        return (
            4
            * np.pi
            * v**2
            * (mass / (2 * np.pi * k_b * temp)) ** (3 / 2)
            * np.exp(-mass * v**2 / (2 * k_b * temp))
        )

    # Use the Maxwell-Boltzmann distribution to sample speeds
    v_max: float = (
        np.sqrt(8 * k_b * temp / (np.pi * mass)) * 5
    )  # upper limit for speeds
    v: NDArray[np.floating[Any]] = np.linspace(0, v_max, 1000)
    cdf = np.cumsum(f(v, temp, mass))
    cdf /= cdf[-1]
    rng = np.random.default_rng()
    random_values = rng.random(n)
    sampled_indices = np.searchsorted(cdf, random_values)
    return v[sampled_indices]


def sample_gaussian_velocities(
    n: int, mu: float, sigma: float
) -> NDArray[np.floating[Any]]:
    # Use numpy to sample from a normal (Gaussian) distribution directly
    rng = np.random.default_rng()
    return rng.normal(loc=mu, scale=sigma, size=n)


@dataclass
class Solenoid:
    """Dataclass representing a solenoid with its parameters."""

    length: float = 1.0
    time_step: float = 0.1
    velocity: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([1.0]))
    init_spin: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    field: NDArray[np.floating[Any]] | Callable[[float], NDArray[np.floating[Any]]] = (
        field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    )
    perp_dist: NDArray[np.floating[Any]] | None = None


class SolenoidSimulator:
    """Main class for running classical solenoid simulations."""

    def __init__(self, solenoid: Solenoid) -> None:
        self.solenoid = solenoid

    def run(
        self,
    ) -> tuple[list[np.ndarray], NDArray[np.floating[Any]]]:
        """Run the spin echo simulation using configured parameters."""
        # Extract parameters with explicit type casting
        velocity = self.solenoid.velocity
        time_step = self.solenoid.time_step
        length = self.solenoid.length
        init_spin = self.solenoid.init_spin
        perp_dist = self.solenoid.perp_dist

        n: int = len(init_spin)
        z_dist = [np.arange(0, length, time_step * v) for v in velocity]

        # --- Custom field support ---
        field_param = self.solenoid.field
        # field_param can be:
        # - a constant vector (list/array of 3)
        # - a callable: field(z) -> array_like shape (3,)
        # - a precomputed array: shape (n, max_steps, 3)

        def get_field(z: float) -> NDArray[np.floating[Any]]:
            if callable(field_param):
                return np.asarray(field_param(z))
            # constant field
            return np.asarray(field_param, dtype=np.float64)

        def off_axis_field_components(
            z: float,
            r: float,
            theta: float,
            dz: float = 1e-5,
        ) -> NDArray[np.floating[Any]]:
            # Get B_0(z) and its derivatives numerically
            b0_z = get_field(z)

            # If B0_z is a vector, take the z-component
            b0_z = np.asarray(b0_z)[2]

            # Get the derivatives numerically
            b0_p = (
                np.asarray(get_field(z + dz))[2] - np.asarray(get_field(z - dz))[2]
            ) / (2 * dz)
            b0_pp = (
                np.asarray(get_field(z + dz))[2]
                - 2 * b0_z
                + np.asarray(get_field(z - dz))[2]
            ) / (dz**2)

            b_r = -0.5 * r * b0_p
            db_z = -0.25 * r**2 * b0_pp

            return np.array([b_r * np.cos(theta), b_r * np.sin(theta), b0_z + db_z])

        # Differential equation dS/dt = gamma * S x B
        def ds_dt(
            s_vec: NDArray[np.floating[Any]], b_vec: NDArray[np.floating[Any]]
        ) -> NDArray[np.floating[Any]]:
            # Check if perp_dist is provided and not all zeros (i.e., at least one particle is off-axis)
            if perp_dist is not None and not np.allclose(perp_dist, 0):
                # perp_dist can be scalar or array; handle both
                r = perp_dist[spin_idx, 0]
                theta = perp_dist[spin_idx, 1]
                b_vec = off_axis_field_components(z_now, r, theta)
            return gyromagnetic_ratio * np.cross(s_vec, b_vec)

        # Find the maximum number of time steps among all spins
        max_steps = max(len(z) for z in z_dist)

        # Initialize spin vector array (3D vectors over time)
        s = np.zeros((n, max_steps, 3))

        # Integrate using 4th-order Runge-Kutta
        for spin_idx in range(n):
            num_steps = len(z_dist[spin_idx])
            s[spin_idx, 0] = init_spin[spin_idx]  # Set initial spin state
            for i in range(num_steps - 1):
                z_now = z_dist[spin_idx][i]
                z_half = (
                    z_dist[spin_idx][i] + 0.5 * (z_dist[spin_idx][i + 1] - z_now)
                    if i + 1 < num_steps
                    else z_now
                )
                # RK4 with position-dependent field
                b1 = get_field(z_now)
                k1 = ds_dt(s[spin_idx, i], b1)
                b2 = get_field(z_half)
                k2 = ds_dt(s[spin_idx, i] + 0.5 * time_step * k1, b2)
                k3 = ds_dt(s[spin_idx, i] + 0.5 * time_step * k2, b2)
                b4 = get_field(z_dist[spin_idx][i + 1] if i + 1 < num_steps else z_now)
                k4 = ds_dt(s[spin_idx, i] + time_step * k3, b4)
                s[spin_idx, i + 1] = s[spin_idx, i] + (time_step / 6.0) * (
                    k1 + 2 * k2 + 2 * k3 + k4
                )

        return z_dist, s
