from __future__ import annotations

from itertools import starmap
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from spinecho_sim.field import FieldRegion, SolenoidRegion
from spinecho_sim.solver import (
    ExperimentalTrajectory,
    FieldSolver,
    MonatomicExperimentalTrajectory,
)
from spinecho_sim.state import (
    CoherentSpin,
    MonatomicParticleState,
    MonatomicTrajectory,
    ParticleDisplacement,
    Spin,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)

if TYPE_CHECKING:
    from spinecho_sim.util import Vec3


def field_vec(field: FieldRegion, z: float, displacement: ParticleDisplacement) -> Vec3:
    """Magnetic field at the particle's transverse displacement at z."""
    x = displacement.x
    y = displacement.y
    return field.field_at(x, y, z)


def simulate_trajectory_cartesian(
    solenoid: SolenoidRegion,
    initial_state: MonatomicParticleState,
    n_steps: int = 100,
) -> ExperimentalTrajectory:
    """Run the spin echo simulation using configured parameters."""
    z_points = np.linspace(0, solenoid.length, n_steps + 1, endpoint=True)

    gyromagnetic_ratio = -2.04e8  # gyromagnetic ratio (rad s^-1 T^-1)

    def _ds_dx(
        z: float,
        spin: tuple[float, float, float],
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        field = field_vec(solenoid, z, initial_state.displacement)
        velocity = initial_state.parallel_velocity

        return (gyromagnetic_ratio / velocity) * np.cross(spin, field)

    sol = solve_ivp(  # type: ignore[return-value]
        fun=_ds_dx,
        t_span=(z_points[0], z_points[-1]),
        y0=initial_state.spin.item(0).cartesian,
        t_eval=z_points,
        vectorized=False,
        rtol=1e-8,
    )
    spins = Spin.from_iter(
        [x.as_generic() for x in starmap(CoherentSpin.from_cartesian, sol.y.T)]  # type: ignore[return-value]
    )
    return MonatomicExperimentalTrajectory(
        trajectory=MonatomicTrajectory(
            _spin_angular_momentum=spins,
            displacement=initial_state.displacement,
            parallel_velocity=initial_state.parallel_velocity,
        ),
        positions=z_points,
    )


def test_simulate_trajectory() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )

    field = SolenoidRegion.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    solver = FieldSolver(region=field)
    n_steps = 300
    result = solver.simulate_monatomic_trajectory(initial_state, n_steps=n_steps)

    assert result.spin.cartesian.shape == (3, n_steps + 1, 1)

    expected = simulate_trajectory_cartesian(field, initial_state, n_steps=n_steps)
    np.testing.assert_allclose(
        result.spin.cartesian,
        expected.spin.cartesian,
        atol=1e-4,
    )


def test_simulate_trajectories() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=2
        ),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )
    field = SolenoidRegion.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    solver = FieldSolver(region=field)
    result = solver.simulate_monatomic_trajectories([initial_state], n_steps=300)
    expected = solver.simulate_monatomic_trajectory(initial_state, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result.spin.theta[0, ..., 0],
        expected.spin.theta[..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.spin.phi[0, ..., 0],
        expected.spin.phi[..., 1],
        atol=1e-4,
    )


def test_simulate_trajectory_high_spin() -> None:
    particle_velocity = 714

    initial_state = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=2
        ),
        displacement=sample_uniform_displacement(1, 1.16e-3)[0],
        parallel_velocity=sample_gaussian_velocities(
            1, particle_velocity, 0.225 * particle_velocity
        )[0],
    )
    field = SolenoidRegion.from_experimental_parameters(
        length=0.75,
        magnetic_constant=3.96e-3,
        current=0.01,
    )
    solver = FieldSolver(region=field)
    result = solver.simulate_monatomic_trajectory(initial_state, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result.spin.theta[0, ..., 0],
        result.spin.theta[1, ..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.spin.phi[0, ..., 0],
        result.spin.phi[1, ..., 1],
        atol=1e-4,
    )

    initial_state_1 = MonatomicParticleState(
        _spin_angular_momentum=CoherentSpin(theta=np.pi / 2, phi=0).as_generic(
            n_stars=1
        ),
        displacement=initial_state.displacement,
        parallel_velocity=initial_state.parallel_velocity,
    )
    result_1 = solver.simulate_monatomic_trajectory(initial_state_1, n_steps=300)

    # Both theta and phi should be the same for all stars
    np.testing.assert_allclose(
        result_1.spin.theta[..., 0],
        result.spin.theta[..., 1],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result_1.spin.phi[..., 0],
        result.spin.phi[..., 1],
        atol=1e-4,
    )
