from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.util import get_figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D

    from spinecho_sim.solenoid._solenoid import SolenoidSimulationResult


def plot_spin_component(
    result: SolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    spins = result.spins.cartesian[:, :, idx]
    average_spins = np.average(spins, axis=0)

    label = [
        r"$\langle S_x \rangle$",
        r"$\langle S_y \rangle$",
        r"$\langle S_z \rangle$",
    ][idx]
    (average_line,) = ax.plot(positions, average_spins, label=label)
    color = average_line.get_color()
    ax.plot(positions, spins.T, alpha=0.1, color=color)

    std_spins = np.std(spins, axis=0)
    ax.fill_between(
        positions,
        average_spins - std_spins,
        average_spins + std_spins,
        alpha=0.2,
        linestyle="--",
        color=color,
    )

    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    ax.set_ylabel("Spin components")

    return fig, ax


def plot_spin_intensity(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    spins = np.average(result.spins.cartesian, axis=0)

    intensity = (spins[:, 0] ** 2 + spins[:, 1] ** 2) / (0.5**2)
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


def plot_spin_components(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(3):
        plot_spin_component(result, idx, ax=ax)

    plot_spin_intensity(result, ax=ax.twinx())
    return fig, ax


def plot_spin_phi(
    result: SolenoidSimulationResult, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)
    positions = result.positions
    phi = result.spins.phi
    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(
        positions,
        average_phi / np.pi,
    )
    color = average_line.get_color()

    ax.plot(
        positions,
        (phi / np.pi),
        linewidth=1.0,
        alpha=0.1,
        color=color,
    )
    std_spins = np.std(phi, axis=0)
    ax.fill_between(
        positions,
        (average_phi - std_spins) / np.pi,
        (average_phi + std_spins) / np.pi,
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    return fig, ax


def plot_spin_theta(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = result.positions
    theta = result.spins.theta
    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(
        positions,
        average_theta / np.pi,
    )
    color = average_line.get_color()

    ax.plot(
        positions,
        (theta / np.pi),
        linewidth=1.0,
        alpha=0.1,
        color=color,
    )
    std_spins = np.std(theta, axis=0)
    ax.fill_between(
        positions,
        (average_theta - std_spins) / np.pi,
        (average_theta + std_spins) / np.pi,
        alpha=0.2,
        linestyle="--",
        color=color,
    )
    return fig, ax


def plot_spin_angles(result: SolenoidSimulationResult) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx in range(3):
        plot_spin_component(result, idx, ax=ax)

    plot_spin_intensity(result, ax=ax.twinx())
    return fig, ax
