"""Core simulation functionality for spin echo experiments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np
from tqdm import tqdm

from spinecho_sim.molecule import diatomic_hamiltonian_dicke
from spinecho_sim.state import (
    EmptySpinList,
    EmptySpinListList,
    MonatomicParticleState,
    MonatomicTrajectory,
    MonatomicTrajectoryList,
    ParticleDisplacement,
    ParticleDisplacementList,
    ParticleState,
    Spin,
    StateVectorParticleState,
    StateVectorTrajectory,
    StateVectorTrajectoryList,
    Trajectory,
    TrajectoryList,
)
from spinecho_sim.util import solve_ivp_typed, sparse_apply, timed, verify_hermitian

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spinecho_sim.field import FieldRegion
    from spinecho_sim.state._state import (
        CoherentMonatomicParticleState,
    )
    from spinecho_sim.util import Vec3


@dataclass(kw_only=True, frozen=True)
class ExperimentalTrajectory:
    """Represents the trajectory of a diatomic particle in a solenoid."""

    trajectory: Trajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spin(self) -> Spin[tuple[int, int]]:
        """The spin components from the simulation states."""
        return self.trajectory.spin

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        """The rotational angular momentum of the particle."""
        return self.trajectory.rotational_angular_momentum

    @property
    def displacement(self) -> ParticleDisplacement:
        """The displacement of the particle at the end of the trajectory."""
        return self.trajectory.displacement


@dataclass(kw_only=True, frozen=True)
class MonatomicExperimentalTrajectory(ExperimentalTrajectory):
    """Represents the trajectory of a monatomic particle in a solenoid."""

    trajectory: MonatomicTrajectory

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        """The rotational angular momentum of the particle."""
        return EmptySpinList(self.spin.shape)


@dataclass(kw_only=True, frozen=True)
class StateVectorExperimentalTrajectory(ExperimentalTrajectory):
    """Represents a trajectory in a solenoid using state vectors instead of spins."""

    trajectory: StateVectorTrajectory
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def state_vectors(self) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """Get the state vectors from the trajectory."""
        return self.trajectory.state_vectors

    @property
    def hilbert_space_dims(self) -> tuple[int, int]:
        """Get the Hilbert space dimensions of the state vectors."""
        return self.trajectory.hilbert_space_dims

    @property
    @override
    def spin(self) -> Spin[tuple[int, int]]:
        msg = "Spin components are not available in state vector representation."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int]]:
        msg = "Rotational angular momentum is not available in state vector representation."
        raise NotImplementedError(msg)

    @staticmethod
    def from_solenoid_trajectory(
        solenoid_trajectory: ExperimentalTrajectory,
        hilbert_space_dims: tuple[int, int],
    ) -> StateVectorExperimentalTrajectory:
        """Create a StateVectorSolenoidTrajectory from a SolenoidTrajectory."""
        return StateVectorExperimentalTrajectory(
            trajectory=StateVectorTrajectory.from_trajectory(
                solenoid_trajectory.trajectory, hilbert_space_dims
            ),
            positions=solenoid_trajectory.positions,
        )


@dataclass(kw_only=True, frozen=True)
class SimulationResult:
    """Represents the result of a solenoid simulation."""

    trajectories: TrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def spin(self) -> Spin[tuple[int, int, int]]:
        return self.trajectories.spin

    @property
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return self.trajectories.rotational_angular_momentum

    @property
    def displacements(self) -> ParticleDisplacementList:
        """Extract the displacements from the simulation states."""
        return self.trajectories.displacements


@dataclass(kw_only=True, frozen=True)
class MonatomicSimulationResult(SimulationResult):
    trajectories: MonatomicTrajectoryList

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        return EmptySpinListList(self.spin.shape)


@dataclass(kw_only=True, frozen=True)
class StateVectorSimulationResult(SimulationResult):
    """Represents the result of a solenoid simulation using state vectors."""

    trajectories: StateVectorTrajectoryList
    positions: np.ndarray[Any, np.dtype[np.floating]]

    @property
    def state_vectors(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        """Get the state vectors from the trajectories."""
        return self.trajectories.state_vectors

    @property
    def hilbert_space_dims(self) -> tuple[int, int]:
        """Get the Hilbert space dimensions of the state vectors."""
        return self.trajectories.hilbert_space_dims

    @property
    @override
    def spin(self) -> Spin[tuple[int, int, int]]:
        msg = "Spin components are not available in state vector representation."
        raise NotImplementedError(msg)

    @property
    @override
    def rotational_angular_momentum(self) -> Spin[tuple[int, int, int]]:
        msg = "Rotational angular momentum is not available in state vector representation."
        raise NotImplementedError(msg)

    @staticmethod
    def from_simulation_result(
        result: SimulationResult,
        hilbert_space_dims: tuple[int, int],
    ) -> StateVectorSimulationResult:
        """Convert a regular SolenoidSimulationResult to a StateVectorSolenoidSimulationResult."""
        # Create state vector trajectories from regular trajectories
        state_vector_trajectories: list[StateVectorTrajectory] = []
        for i in range(len(result.trajectories)):
            trajectory = result.trajectories[i]
            sv_trajectory = StateVectorTrajectory.from_trajectory(
                trajectory, hilbert_space_dims
            )
            state_vector_trajectories.append(sv_trajectory)

        return StateVectorSimulationResult(
            trajectories=StateVectorTrajectoryList.from_state_vector_trajectories(
                state_vector_trajectories
            ),
            positions=result.positions,
        )


@dataclass(kw_only=True, frozen=True)
class FieldSolver:
    """Dataclass representing a solenoid with its parameters."""

    region: FieldRegion
    z_start: float | None = None
    z_end: float | None = None

    @property
    def z_span(self) -> tuple[float, float]:
        if self.z_start is not None and self.z_end is not None:
            return (self.z_start, self.z_end)
        ext = self.region.extent
        if ext is None:
            msg = "Region has no finite extent; supply z_start and z_end."
            raise ValueError(msg)
        return (ext.z[0], ext.z[1])

    @property
    def length(self) -> float:
        z0, z1 = self.z_span
        return float(z1 - z0)

    def _field_vec(self, z: float, displacement: ParticleDisplacement) -> Vec3:
        """Magnetic field at the particle's transverse displacement at z."""
        x = displacement.x
        y = displacement.y
        return self.region.field_at(x, y, z)

    def simulate_diatomic_trajectory(
        self,
        initial_state: StateVectorParticleState,
        n_steps: int = 100,
    ) -> StateVectorExperimentalTrajectory:
        i = (initial_state.hilbert_space_dims[0] - 1) / 2
        j = (initial_state.hilbert_space_dims[1] - 1) / 2  # dim=2j+1
        assert i > 1 / 2, "Invalid diatomic spin state: i must be > 1/2"
        assert j > 1 / 2, "Invalid diatomic spin state: j must be > 1/2"

        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        def schrodinger_eq(z: float, psi: np.ndarray) -> np.ndarray:
            field = self._field_vec(z, initial_state.displacement)
            hamiltonian = diatomic_hamiltonian_dicke(
                i, j, initial_state.coefficients, field
            )
            assert verify_hermitian(hamiltonian), "Hamiltonian is not Hermitian"
            result = sparse_apply(hamiltonian, psi)
            psi /= np.linalg.norm(psi)  # Ensures state remains normalized
            return -1j * result / initial_state.parallel_velocity

        psi0 = initial_state.state_vector.copy()

        sol = solve_ivp_typed(
            fun=schrodinger_eq,
            t_span=(z_points[0], z_points[-1]),
            y0=psi0,
            t_eval=z_points,
            rtol=1e-8,
        )

        state_vectors: np.ndarray[tuple[int, int], np.dtype[np.complex128]] = (
            np.transpose(sol.y).astype(np.complex128)
        )

        for idx, psi in enumerate(state_vectors):
            norm = np.linalg.norm(psi)
            if not np.isclose(norm, 1.0, atol=1e-8):
                warnings.warn(
                    f"State vector at index {idx} is not normalized: norm = {norm}",
                    UserWarning,
                    stacklevel=2,
                )

        return StateVectorExperimentalTrajectory(
            trajectory=StateVectorTrajectory(
                state_vectors=state_vectors,
                hilbert_space_dims=initial_state.hilbert_space_dims,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    @timed
    def simulate_diatomic_trajectories(
        self,
        initial_states: Sequence[StateVectorParticleState],
        n_steps: int = 100,
    ) -> StateVectorSimulationResult:
        """Run a solenoid simulation for multiple initial states."""
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)
        return StateVectorSimulationResult(
            trajectories=StateVectorTrajectoryList.from_state_vector_trajectories(
                [
                    self.simulate_diatomic_trajectory(state, n_steps).trajectory
                    for state in tqdm(initial_states, desc="Simulating Trajectories")
                ]
            ),
            positions=z_points,
        )

    def _simulate_coherent_monatomic_trajectory(
        self,
        initial_state: CoherentMonatomicParticleState,
        n_steps: int = 100,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.floating]],
        np.ndarray[Any, np.dtype[np.floating]],
    ]:
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        gyromagnetic_ratio = initial_state.gyromagnetic_ratio
        effective_ratio = gyromagnetic_ratio / initial_state.parallel_velocity

        def _d_angles_dx(
            z: float, angles: tuple[float, float]
        ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
            theta, phi = angles
            # TODO: can we find B_phi and B_theta analytically to make this faster?  # noqa: FIX002
            field = self._field_vec(z, initial_state.displacement)

            # Ensure theta is not too close to 0 or pi to avoid coordinate singularity
            epsilon = 1e-12
            if np.abs(theta) < epsilon:
                theta = epsilon
            elif np.abs(theta - np.pi) < epsilon:
                theta = np.pi - epsilon

            # d_theta / dt = B_x sin phi - B_y cos phi
            d_theta = field[0] * np.sin(phi) - field[1] * np.cos(phi)
            # d_phi / dt = tan theta * (B_x cos phi + B_y sin phi) - B_z
            d_phi_xy = (field[0] * np.cos(phi) + field[1] * np.sin(phi)) / np.tan(theta)
            d_phi = d_phi_xy - field[2]
            return effective_ratio * np.array([d_theta, d_phi])

        y0 = np.array(
            [
                initial_state.spin.theta.item(),
                initial_state.spin.phi.item(),
            ]
        )

        sol = solve_ivp_typed(
            fun=_d_angles_dx,  # pyright: ignore[reportArgumentType]
            t_span=(z_points[0], z_points[-1]),
            y0=y0,
            t_eval=z_points,
            vectorized=False,
            rtol=1e-8,
        )
        return sol.y[0], sol.y[1]

    def simulate_monatomic_trajectory(
        self,
        initial_state: ParticleState,
        n_steps: int = 100,
    ) -> MonatomicExperimentalTrajectory:
        """Run the spin echo simulation using configured parameters."""
        assert isinstance(initial_state, MonatomicParticleState), (
            "Expected a coherent monatomic particle state."
        )

        data = np.empty((n_steps + 1, initial_state.spin.size, 2), dtype=np.float64)
        for i, s in enumerate(initial_state.as_coherent()):
            thetas, phis = self._simulate_coherent_monatomic_trajectory(s, n_steps)
            data[:, i, 0] = thetas
            data[:, i, 1] = phis

        spins = Spin[tuple[int, int]](data)
        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)

        return MonatomicExperimentalTrajectory(
            trajectory=MonatomicTrajectory(
                _spin_angular_momentum=spins,
                displacement=initial_state.displacement,
                parallel_velocity=initial_state.parallel_velocity,
            ),
            positions=z_points,
        )

    @timed
    def simulate_monatomic_trajectories(
        self,
        initial_states: Sequence[ParticleState],
        n_steps: int = 100,
    ) -> MonatomicSimulationResult:
        """Run a solenoid simulation for multiple initial states."""
        mono_initial_states = [
            state
            for state in initial_states
            if isinstance(state, MonatomicParticleState)
        ]
        assert mono_initial_states, "No MonatomicParticleState instances provided."

        z_points = np.linspace(0, self.length, n_steps + 1, endpoint=True)
        return MonatomicSimulationResult(
            trajectories=MonatomicTrajectoryList.from_monatomic_trajectories(
                [
                    self.simulate_monatomic_trajectory(state, n_steps).trajectory
                    for state in tqdm(
                        mono_initial_states, desc="Simulating Trajectories"
                    )
                ]
            ),
            positions=z_points,
        )
