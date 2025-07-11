"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

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
    dist_step: float = 0.1
    velocity: NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([1.0]))
    init_spin: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    field: Callable[[float], NDArray[np.floating[Any]]] = field(
        default_factory=lambda: (lambda z: np.array([0.0, 0.0, 1.0]))
    )
    perp_dist: NDArray[np.floating[Any]] | None = None


class SolenoidSimulator:
    """Main class for running classical solenoid simulations."""

    def __init__(self, solenoid: Solenoid) -> None:
        self.solenoid = solenoid

    def run(
        self,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Run the spin echo simulation using configured parameters."""
        # Extract parameters with explicit type casting
        init_spin = self.solenoid.init_spin
        n: int = len(init_spin)
        z_dist = np.arange(0, self.solenoid.length, self.solenoid.dist_step)
        steps = len(z_dist)

        # Initialize spin vector array (3D vectors over time)
        s = np.zeros((n, steps, 3))

        # Integrate using 4th-order Runge-Kutta
        for spin_idx in range(n):
            s[spin_idx, 0] = init_spin[spin_idx]  # Set initial spin state

            velocity: float = float(self.solenoid.velocity[spin_idx])
            # Integrate using solve_ivp
            sol = solve_ivp(
                fun=lambda z, s_vec: _dsdx(z, s_vec, self.solenoid, velocity, spin_idx),
                t_span=(z_dist[0], z_dist[-1]),
                y0=init_spin[spin_idx],
                t_eval=z_dist,
                method="RK45",  # or "RK23", "DOP853", etc.
                vectorized=False,
                rtol=1e-8,
                # atol=1e-10,
            )
            s[spin_idx, :, :] = sol.y.T  # shape (steps, 3)

        return z_dist, s


def _get_field(
    z: float,
    solenoid: Solenoid,
) -> NDArray[np.floating[Any]]:
    return np.asarray(solenoid.field(z))


def _off_axis_field(
    z: float,
    r: float,
    theta: float,
    solenoid: Solenoid,
    dz: float = 1e-5,
) -> NDArray[np.floating[Any]]:
    # Get B_0(z) and its derivatives numerically
    b0_z = _get_field(z, solenoid)[2]  # Get the z-component of the field

    # Sample points for finite difference
    z_points = np.array([z - dz, z, z + dz])
    b_z_values = np.array([_get_field(zi, solenoid)[2] for zi in z_points])

    # Use numpy.gradient for first and second derivatives
    b0_p = np.gradient(b_z_values, dz)[1]  # First derivative at z
    b0_pp = np.gradient(np.gradient(b_z_values, dz), dz)[1]  # Second derivative at z

    b0_z = b_z_values[1]

    b_r = -0.5 * r * b0_p
    db_z = -0.25 * r**2 * b0_pp

    return np.array([b_r * np.cos(theta), b_r * np.sin(theta), b0_z + db_z])


# Differential equation dS/dt = gamma * S x B
def _dsdx(
    z: float,
    s_vec: NDArray[np.floating[Any]],
    solenoid: Solenoid,
    velocity: float,
    spin_idx: int,
) -> NDArray[np.floating[Any]]:
    b_vec = _get_field(z, solenoid)
    # Optionally handle off-axis field here if needed
    if solenoid.perp_dist is not None and not np.allclose(solenoid.perp_dist, 0):
        r = solenoid.perp_dist[spin_idx, 0]
        theta = solenoid.perp_dist[spin_idx, 1]
        b_vec = _off_axis_field(z, r, theta, solenoid)
    return (gyromagnetic_ratio / velocity) * np.cross(s_vec, b_vec)
