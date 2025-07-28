from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.util import Measure, get_figure, plot_measure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from spinecho_sim.solenoid._solenoid import SolenoidSimulationResult

cambridge_core_blue = (0 / 255, 115 / 255, 207 / 255)
cambridge_core_orange = (227 / 255, 114 / 255, 34 / 255)


def plot_spin_component_old(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    spins = result.spins.cartesian[idx]
    average_spins = np.average(spins, axis=0)

    label = [
        r"$\langle S_x \rangle$",
        r"$\langle S_y \rangle$",
        r"$\langle S_z \rangle$",
    ][idx]
    (average_line,) = ax.plot(positions, average_spins, label=label)
    color = average_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(spins, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(spins, axis=0) / np.sqrt(len(spins))
    ax.fill_between(
        positions,
        np.clip(average_spins - std_spins, -1, 1).ravel(),
        np.clip(average_spins + std_spins, -1, 1).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin components")

    return fig, ax


def plot_spin_intensity_old(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    intensity = np.average(
        result.spins.cartesian[0] ** 2 + result.spins.cartesian[1] ** 2, axis=0
    )
    (line,) = ax.plot(
        positions,
        intensity,
        label=r"$I_\parallel =\langle S_x\rangle^2+\langle S_y\rangle^2$",
        color="black",
        linestyle="--",
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin Intensity")

    return fig, ax, line


def plot_spin_components_old(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    assert result.spins.n_stars == 1, "Component plots only supports spin-1/2 systems"
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(3):
        plot_spin_component_old(result, idx, ax=ax)

    plot_spin_intensity_old(result, ax=ax.twinx())
    return fig, ax


def plot_spin_phi(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)
    positions = result.positions
    phi = np.unwrap(result.spins.phi, axis=1) / np.pi
    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(positions, average_phi)
    color = average_line.get_color()
    average_line.set_label(r"$\langle \phi \rangle$")

    ax.plot(
        positions,
        np.swapaxes(phi, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(phi, axis=0) / np.sqrt(len(phi))
    ax.fill_between(
        positions,
        (average_phi - std_spins).ravel(),
        (average_phi + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel(r"Phase Angles (radians/$\pi$)")
    return fig, ax


def plot_spin_theta(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)
    positions = result.positions
    theta = np.unwrap(result.spins.theta, axis=1) / np.pi
    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(
        positions,
        average_theta,
    )
    color = average_line.get_color()
    average_line.set_label(r"$\langle \theta \rangle$")

    ax.plot(
        positions,
        np.swapaxes(theta, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(theta, axis=0) / np.sqrt(len(theta))
    ax.fill_between(
        positions,
        (average_theta - std_spins).ravel(),
        (average_theta + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel(r"Phase Angles (radians/$\pi$)")
    return fig, ax


def plot_spin_angles(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    assert result.spins.n_stars == 1, "Component plots only supports spin-1/2 systems"
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_spin_theta(result, ax=ax)
    plot_spin_phi(result, ax=ax)

    return fig, ax


def plot_spin_state(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spins.momentum_states[idx, :, :]

    average_state_measure = np.average(plot_measure(states, measure)[0], axis=0)

    n_stars = result.spins.n_stars
    s = n_stars / 2
    ms_values = np.linspace(s, -s, n_stars + 1, endpoint=True)
    ms_labels = [
        rf"$|m_S={int(m)} \rangle$"
        if m.is_integer()
        else rf"$|m_S={2 * m:.0f}/2 \rangle$"
        for m in ms_values
    ]

    # Plot phase
    (measure_line,) = ax.plot(
        positions,
        average_state_measure,
        label="Mean",
        color=cambridge_core_blue,
    )
    color_measure = measure_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(plot_measure(states, measure)[0], 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_measure,
    )

    # Standard error of the mean for phase
    std_states_measure = np.std(plot_measure(states, measure)[0], axis=0) / np.sqrt(
        len(states)
    )
    ax.fill_between(
        positions,
        (average_state_measure - std_states_measure).ravel(),
        (average_state_measure + std_states_measure).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_measure,
        label=r"Mean $\pm 1\sigma$",
    )

    ax.set_ylabel(f"{ms_labels[idx]} {plot_measure(states, measure)[1]}")
    ax.legend(loc="lower right")
    ax.set_xlim(positions[0], positions[-1])

    return fig, ax


def plot_state_intensity(
    result: SolenoidSimulationResult, idx: int, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spins.momentum_states[idx]
    average_state_abs = np.average(np.abs(states) ** 2, axis=0)

    (line,) = ax.plot(
        positions,
        average_state_abs,
        color="black",
        linestyle="--",
        label=r"$|m_S\rangle$ Intensity",
    )
    ax.set_ylabel(r"$|m_S\rangle$ Intensity")
    ax.set_xlim(positions[0], positions[-1])
    ax.legend(loc="center right")

    return fig, ax, line


def plot_spin_states(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    n_stars = result.spins.n_stars
    fig, axes = plt.subplots(n_stars + 1, 2, figsize=(10, 6), sharex=True)

    for idx, (ax_abs, ax_arg) in enumerate(axes):
        plot_spin_state(result, idx, measure="abs", ax=ax_abs)
        plot_spin_state(result, idx, measure="arg", ax=ax_arg)
    for ax in axes[-1]:
        ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_expectation_value(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = result.spin_expectations[idx, :]

    average_state_measure = np.average(expectation_values, axis=0)
    labels = [
        r"\langle S_x \rangle",
        r"\langle S_y \rangle",
        r"\langle S_z \rangle",
    ]

    # Plot phase
    (measure_line,) = ax.plot(
        positions,
        average_state_measure,
        label=rf"$\overline{{{labels[idx]}}} / \hbar$",
        color=cambridge_core_blue,
    )
    color_measure = measure_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(expectation_values, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_measure,
    )

    # Standard error of the mean for phase
    std_states_measure = np.std(expectation_values, axis=0) / np.sqrt(
        len(expectation_values)
    )
    ax.fill_between(
        positions,
        (average_state_measure - std_states_measure).ravel(),
        (average_state_measure + std_states_measure).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_measure,
        label=rf"$\overline{{{labels[idx]}}} / \hbar \pm 1\sigma$",
    )

    ax.set_ylabel(rf"${labels[idx]} / \hbar$")
    ax.legend(loc="center left")
    ax.set_xlim(positions[0], positions[-1])

    return fig, ax


def plot_expectation_values(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for idx, ax in enumerate(axes):
        plot_expectation_value(result, idx, ax=ax)
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_expectation_phi(
    result: SolenoidSimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = result.spin_expectations

    wrapped_phi = np.arctan2(
        expectation_values[1, :], expectation_values[0, :]
    )  # atan2(y, x) gives the angle in radians
    phi = np.unwrap(wrapped_phi, axis=1) / np.pi  # Unwrap and normalize to [0, 2π)

    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(
        positions,
        average_phi,
        label=r"$\overline{\langle \phi \rangle}$",
        color=cambridge_core_blue,
    )
    color = average_line.get_color()

    ax.plot(
        positions,
        np.swapaxes(phi, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(phi, axis=0) / np.sqrt(len(phi))
    ax.fill_between(
        positions,
        (average_phi - std_spins).ravel(),
        (average_phi + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
        label=r"$\overline{\langle \phi \rangle} \pm 1 \sigma$",
    )
    ax.legend(loc="lower right")
    ax.set_ylabel(r"$\langle \phi \rangle$ Azimuthal Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_expectation_theta(
    result: SolenoidSimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = result.spin_expectations

    wrapped_theta = np.arctan2(
        np.sqrt(expectation_values[0, :] ** 2 + expectation_values[1, :] ** 2),
        expectation_values[2, :],
    )
    theta = np.unwrap(wrapped_theta, axis=1) / np.pi

    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(
        positions,
        average_theta,
        label=r"$\overline{\langle \theta \rangle}$",
        color=cambridge_core_orange,
    )
    color = average_line.get_color()

    ax.plot(
        positions,
        np.swapaxes(theta, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color,
    )
    # Standard error of the mean
    std_spins = np.std(theta, axis=0) / np.sqrt(len(theta))
    ax.fill_between(
        positions,
        (average_theta - std_spins).ravel(),
        (average_theta + std_spins).ravel(),
        alpha=0.2,
        linestyle="--",
        color=color,
        label=r"$\overline{\langle \theta \rangle} \pm 1 \sigma$",
    )
    ax.legend(loc="upper right")
    ax.set_ylabel(r"$\langle \theta \rangle$ Polar Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_expectation_angles(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_expectation_theta(result, ax=ax)
    plot_expectation_phi(result, ax=ax.twinx())
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title

    return fig, ax


def plot_expectation_trajectory_3d(
    result: SolenoidSimulationResult,
) -> tuple[Figure, Axes3D]:
    fig = plt.figure(figsize=(6, 6))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))

    expectations = result.spin_expectations  # shape: (3, n_samples, n_positions)
    # Average over samples (axis=1), shape: (3, n_positions)
    avg_expectations = np.average(expectations, axis=1)

    # Unpack components
    x = avg_expectations[0, :]
    y = avg_expectations[1, :]
    z = avg_expectations[2, :]

    # Plot the trajectory as a 3D curve
    (average_line,) = ax.plot(
        x,
        y,
        z,
        label=r"$\overline{\langle \mathbf{S} \rangle}$",
        color=cambridge_core_blue,
    )
    color = average_line.get_color()
    ax.plot(
        np.swapaxes(expectations[0], 0, 1).reshape(expectations[0].size, -1),
        np.swapaxes(expectations[1], 0, 1).reshape(expectations[1].size, -1),
        np.swapaxes(expectations[2], 0, 1).reshape(expectations[2].size, -1),
        alpha=0.1,
        color=color,
    )

    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, ax
