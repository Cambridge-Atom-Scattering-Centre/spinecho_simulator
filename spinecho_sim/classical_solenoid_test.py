"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

gyromagnetic_ratio = 2.0 * np.pi * 1.0  # gyromagnetic ratio (rad s^-1 T^-1)


# -- helper: sample N uniform directions on the unit sphere --
def sample_unit_sphere(n: int) -> NDArray[np.floating[Any]]:
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
def sample_unit_circle(n: int) -> NDArray[np.floating[Any]]:
    """Sample N uniform random directions on the unit circle normal to z-axis.

    Returns
    -------
    NDArray[np.floating[Any]]
        An array of shape (N, 3) containing unit vectors.
    """
    phi: NDArray[np.floating[Any]] = np.random.uniform(0, 2 * np.pi, size=n)
    return np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)


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
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Run the spin echo simulation using configured parameters."""
        # Extract parameters with explicit type casting
        velocity: float = float(self.parameters.get("velocity", 1.0))
        field: NDArray[np.floating[Any]] = np.array(
            self.parameters.get("field", [0.0, 0.0, 1.0])
        )
        time_step: float = float(self.parameters.get("time_step", 0.1))
        length: float = float(self.parameters.get("length", 1.0))
        init_spin: NDArray[np.floating[Any]] = np.array(
            self.parameters.get("init_spin", [1.0, 0.0, 0.0])
        )

        n: int = len(init_spin)

        # Time array based on velocity and length
        t: NDArray[np.floating[Any]] = np.arange(0, length / velocity, time_step)

        # Initialize spin vector array (3D vectors over time)
        s: NDArray[np.floating[Any]] = np.zeros((len(t), n, 3))
        s[0] = init_spin  # Set initial spin state

        # Differential equation dS/dt = gamma * S x B
        def ds_dt(s_vec: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
            return gyromagnetic_ratio * np.cross(s_vec, field)

        # Integrate using 4th-order Runge-Kutta
        for i in range(len(t) - 1):
            k1 = ds_dt(s[i])
            k2 = ds_dt(s[i] + 0.5 * time_step * k1)
            k3 = ds_dt(s[i] + 0.5 * time_step * k2)
            k4 = ds_dt(s[i] + time_step * k3)
            s[i + 1] = s[i] + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t, s
