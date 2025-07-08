"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

gyromagnetic_ratio = 2.0 * np.pi * 1.0  # gyromagnetic ratio (rad s^-1 T^-1)


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


class SolenoidSimulator:
    """Main class for running classical solenoid simulations."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        """Initialize simulator with parameters.

        Args:
            parameters: Dictionary containing simulation parameters
        """
        self.parameters = parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate required parameters are present.

        Raises
        ------
        ValueError
            If any required parameter is missing.
        """
        required_params = [
            "velocity",
            "field",
            "time_step",
            "length",
            "init_spin",
        ]
        missing = [p for p in required_params if p not in self.parameters]
        if missing:
            msg = f"Missing required parameters: {missing}"
            raise ValueError(msg)

    def run(
        self,
    ) -> tuple[list[np.ndarray], NDArray[np.floating[Any]]]:
        """Run the spin echo simulation using configured parameters."""
        # Extract parameters with explicit type casting
        velocity: NDArray[np.floating[Any]] = np.array(
            self.parameters.get("velocity", 1.0)
        )
        field: NDArray[np.floating[Any]] = np.array(
            self.parameters.get("field", [0.0, 0.0, 1.0])
        )
        time_step: float = float(self.parameters.get("time_step", 0.1))
        length: float = float(self.parameters.get("length", 1.0))
        init_spin: NDArray[np.floating[Any]] = np.array(
            self.parameters.get("init_spin", [1.0, 0.0, 0.0])
        )

        n: int = len(init_spin)

        # Distance and time arrays based on velocity and length
        # Create a separate distance array for each particle with its own velocity
        dist = [np.arange(0, length, time_step * v) for v in velocity]
        [
            d / v for d, v in zip(dist, velocity, strict=False)
        ]  # time array for each particle

        # Differential equation dS/dt = gamma * S x B
        def ds_dt(s_vec: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return gyromagnetic_ratio * np.cross(s_vec, field)

        # Find the maximum number of time steps among all spins
        max_steps = max(len(d) for d in dist)

        # Initialize spin vector array (3D vectors over time)
        s: NDArray[np.floating[Any]] = np.zeros((n, max_steps, 3))

        # Integrate using 4th-order Runge-Kutta
        for spin_idx in range(n):
            num_steps = len(dist[spin_idx])
            s[spin_idx, 0] = init_spin[spin_idx]  # Set initial spin state
            for i in range(num_steps - 1):
                k1 = ds_dt(s[spin_idx, i])
                k2 = ds_dt(s[spin_idx, i] + 0.5 * time_step * k1)
                k3 = ds_dt(s[spin_idx, i] + 0.5 * time_step * k2)
                k4 = ds_dt(s[spin_idx, i] + time_step * k3)
                s[spin_idx, i + 1] = s[spin_idx, i] + (time_step / 6.0) * (
                    k1 + 2 * k2 + 2 * k3 + k4
                )

        return dist, s
