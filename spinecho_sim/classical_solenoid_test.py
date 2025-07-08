"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

gyromagnetic_ratio = 2.0 * np.pi * 1.0  # gyromagnetic ratio (rad s^-1 T^-1)


class SolenoidSimulator:
    """Main class for running classical solenoid simulations."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        """Initialize simulator with parameters.

        Args:
            parameters: Dictionary containing simulation parameters like
                'field_strength', 'frequency', 'duration', etc.
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

        # Time array based on velocity and length
        t: NDArray[np.floating[Any]] = np.arange(0, length / velocity, time_step)

        # Initialize spin vector array (3D vectors over time)
        s: NDArray[np.floating[Any]] = np.zeros((len(t), 3))
        s[0] = init_spin  # Set initial spin state

        # Simple spin echo simulation (placeholder)
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

        # Return the x-component as the observable signal
        # signal: NDArray[np.floating[Any]] = s[:, 0]
        return t, s
