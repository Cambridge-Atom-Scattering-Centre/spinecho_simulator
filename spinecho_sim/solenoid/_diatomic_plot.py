from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.molecule.hamiltonian_dicke import collective_ops_sparse
from spinecho_sim.util import get_figure, sparse_apply

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

    from spinecho_sim.solenoid._solenoid import (
        StateVectorSolenoidSimulationResult,
    )


def plot_monatomic_expectation_value(
    result: StateVectorSolenoidSimulationResult,
    idx: int,
    spin: Literal["I", "J"],
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    if spin == "I":
        ops, _ = collective_ops_sparse(i, j)
    else:
        _, ops = collective_ops_sparse(i, j)

    if spin == "I":
        labels = [
            r"\langle I_x \rangle",
            r"\langle I_y \rangle",
            r"\langle I_z \rangle",
        ]
    else:
        labels = [
            r"\langle J_x \rangle",
            r"\langle J_y \rangle",
            r"\langle J_z \rangle",
        ]

    positions = result.positions
    state_vectors = (
        result.state_vectors
    )  # Shape: [number of particles, positions, components]

    # Initialize an array to store expectation values
    expectation_values = np.zeros(
        (state_vectors.shape[0], state_vectors.shape[1]), dtype=np.float64
    )

    # Iterate over particles and positions
    for particle_idx in range(state_vectors.shape[0]):
        for position_idx in range(state_vectors.shape[1]):
            state = state_vectors[
                particle_idx, position_idx, :
            ]  # Extract the state vector
            expectation_values[particle_idx, position_idx] = np.real(
                np.vdot(state, sparse_apply(ops[idx], state))
            )

    average_state_measure = np.average(expectation_values, axis=0)

    (measure_line,) = ax.plot(positions, average_state_measure)
    measure_line.set_label(rf"$\overline{{{labels[idx]}}} / \hbar$")
    color_measure = measure_line.get_color()
    for particle_idx in range(expectation_values.shape[0]):
        ax.plot(
            positions,
            expectation_values[particle_idx, :],
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


def plot_diatomic_expectation_values(
    result: StateVectorSolenoidSimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)

    for idx in range(3):  # Iterate over the three expectation values (x, y, z)
        # Plot I expectation values in the first column
        plot_monatomic_expectation_value(result, idx, "I", ax=axes[idx, 0])
        axes[idx, 0].set_title(rf"$\langle I_{['x', 'y', 'z'][idx]} \rangle$")

        # Plot J expectation values in the second column
        plot_monatomic_expectation_value(result, idx, "J", ax=axes[idx, 1])
        axes[idx, 1].set_title(rf"$\langle J_{['x', 'y', 'z'][idx]} \rangle$")

    # Set shared x-axis label for the bottom row
    axes[-1, 0].set_xlabel(r"Distance $z$ along Solenoid Axis")
    axes[-1, 1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_diatomic_normalisation(
    result: StateVectorSolenoidSimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    for particle_idx in range(result.state_vectors.shape[0]):
        state = result.state_vectors[particle_idx]
        norm = np.linalg.norm(state, axis=1)
        axes.plot(result.positions, norm)

    axes.set_ylabel(r"Normalization $\|\psi\|$")
    axes.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes
