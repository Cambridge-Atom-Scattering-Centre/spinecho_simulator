from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from spinecho_sim.state import (
    CoherentSpin,
    MonatomicTrajectory,
    MonatomicTrajectoryList,
    ParticleDisplacement,
    Spin,
    get_expectation_values,
)


@pytest.mark.parametrize(
    ("state_coefficients", "expected_jx", "expected_jy", "expected_jz"),
    # order of state components is m_s=[j, j-1, ..., -j+1, -j]
    [
        (np.array([1, 0], dtype=np.complex128), 0.0, 0.0, 0.5),  # |+z>
        (np.array([0, 1], dtype=np.complex128), 0.0, 0.0, -0.5),  # |-z>
        (np.array([1, 1], dtype=np.complex128) / np.sqrt(2), -0.5, 0.0, 0.0),  # |+x>
        (np.array([1, -1], dtype=np.complex128) / np.sqrt(2), 0.5, 0.0, 0.0),  # |-x>
        (np.array([1, 1j], dtype=np.complex128) / np.sqrt(2), 0.0, -0.5, 0.0),  # |+y>
        (np.array([1, -1j], dtype=np.complex128) / np.sqrt(2), 0.0, 0.5, 0.0),  # |-y>
        (np.array([1, 0, 0], dtype=np.complex128), 0.0, 0.0, 1.0),  # |z:+1>
        (np.array([0, 0, 1], dtype=np.complex128), 0.0, 0.0, -1.0),  # |z:-1>
        (np.array([0, 1, 0], dtype=np.complex128), 0.0, 0.0, 0.0),  # |z:0>
        (
            np.array([1, np.sqrt(2), 1], dtype=np.complex128) / 2,
            -1.0,
            0.0,
            0.0,
        ),  # |x:+1>
        (
            np.array([-1, 0, 1], dtype=np.complex128) / np.sqrt(2),
            0.0,
            0.0,
            0.0,
        ),  # |x:0>
        (
            np.array([1, -np.sqrt(2), 1], dtype=np.complex128) / 2,
            1.0,
            0.0,
            0.0,
        ),  # |x:-1>
        (
            np.array([1, 1j * np.sqrt(2), -1], dtype=np.complex128) / 2,
            0,
            -1,
            0.0,
        ),  # |y:+1>
        (np.array([1, 0, 1], dtype=np.complex128) / np.sqrt(2), 0, 0, 0.0),  # |y:0>
        (
            np.array([1, -1j * np.sqrt(2), -1], dtype=np.complex128) / 2,
            0,
            1,
            0.0,
        ),  # |y:-1>
    ],
)
def test_expectation_of_known_states(
    state_coefficients: np.ndarray[Any, np.dtype[np.complex128]],
    expected_jx: float,
    expected_jy: float,
    expected_jz: float,
) -> None:
    jx, jy, jz = get_expectation_values(Spin.from_momentum_state(state_coefficients))
    np.testing.assert_array_almost_equal(
        jx,
        expected_jx,
        err_msg=f"Failed for state {state_coefficients}, expected Jx={expected_jx}, got Jx={jx}",
    )
    np.testing.assert_array_almost_equal(
        jy,
        expected_jy,
        err_msg=f"Failed for state {state_coefficients}, expected Jy={expected_jy}, got Jy={jy}",
    )
    np.testing.assert_array_almost_equal(
        jz,
        expected_jz,
        err_msg=f"Failed for state {state_coefficients}, expected Jz={expected_jz}, got Jz={jz}",
    )


@pytest.mark.parametrize("n_stars", [1, 2, 3, 4, 5])
def test_expectation_coherent_state(n_stars: int) -> None:
    rng = np.random.default_rng()
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    spin = CoherentSpin(theta=theta, phi=phi)

    generic_spin = spin.as_generic(n_stars=n_stars)
    expectation = get_expectation_values(generic_spin)

    jx, jy, jz = 2 * expectation / n_stars
    np.testing.assert_array_almost_equal(jx, spin.x, err_msg="Incorrect Jx")
    np.testing.assert_array_almost_equal(jy, spin.y, err_msg="Incorrect Jy")
    np.testing.assert_array_almost_equal(jz, spin.z, err_msg="Incorrect Jz")


@pytest.mark.parametrize("n_stars", [1, 2, 3, 4, 5])
def test_expectation_large_state(n_stars: int) -> None:
    rng = np.random.default_rng()
    theta = rng.uniform(0, np.pi, (1, 1001))
    phi = rng.uniform(0, 2 * np.pi, (1, 1001))
    spins = np.stack(
        [
            np.repeat(theta[..., np.newaxis], n_stars, axis=-1),
            np.repeat(phi[..., np.newaxis], n_stars, axis=-1),
        ],
        axis=-1,
    )
    spin = Spin(spins)
    assert spin.n_stars == n_stars

    expectation = get_expectation_values(spin)  # type: ignore[return-value]

    jx, jy, jz = 2 * expectation / n_stars
    np.testing.assert_array_almost_equal(jx, spin.x[..., 0], err_msg="Incorrect Jx")
    np.testing.assert_array_almost_equal(jy, spin.y[..., 0], err_msg="Incorrect Jy")
    np.testing.assert_array_almost_equal(jz, spin.z[..., 0], err_msg="Incorrect Jz")


def test_spin_from_iter() -> None:
    spins = [
        CoherentSpin(theta=np.pi / 2, phi=0).as_generic(n_stars=2),
        CoherentSpin(theta=np.pi / 3, phi=np.pi / 4).as_generic(n_stars=2),
    ]
    assert spins[0].shape == (2,)

    from_iter = Spin.from_iter(spins)
    assert from_iter.shape == (2, 2)

    from_iter_deep = Spin.from_iter([from_iter])
    assert from_iter_deep.shape == (1, 2, 2)

    np.testing.assert_array_equal(
        from_iter.theta[..., 0],
        from_iter_deep.theta[0, ..., 0],
        err_msg="Theta values do not match",
    )
    np.testing.assert_array_equal(
        from_iter.phi[..., 0],
        from_iter_deep.phi[0, ..., 0],
        err_msg="Phi values do not match",
    )


def test_trajectory_list() -> None:
    n_stars = 2
    spins = [
        CoherentSpin(theta=np.pi / 2, phi=0).as_generic(n_stars=n_stars),
        CoherentSpin(theta=np.pi / 3, phi=np.pi / 4).as_generic(n_stars=n_stars),
    ]
    trajectory = MonatomicTrajectory(
        _spin_angular_momentum=Spin.from_iter(spins),
        displacement=ParticleDisplacement(r=0, theta=0),
        parallel_velocity=10.0,
    )
    trajectory_list = MonatomicTrajectoryList.from_trajectories([trajectory])

    assert len(trajectory_list) == 1
    assert trajectory_list.spin.n_stars == n_stars

    np.testing.assert_array_equal(
        trajectory_list.spin.theta[0, ..., 0],
        trajectory.spin.theta[..., 0],
        err_msg="Theta values do not match",
    )
    np.testing.assert_array_equal(
        trajectory_list.spin.phi[0, ..., 0],
        trajectory.spin.phi[..., 0],
        err_msg="Phi values do not match",
    )
