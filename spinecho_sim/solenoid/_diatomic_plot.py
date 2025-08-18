from __future__ import annotations

from typing import TYPE_CHECKING

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
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    i_ops, _ = collective_ops_sparse(i, j)

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
                np.vdot(state, sparse_apply(i_ops[idx], state))
            )

    average_state_measure = np.average(expectation_values, axis=0)
    labels = [
        r"\langle I_x \rangle",
        r"\langle I_y \rangle",
        r"\langle I_z \rangle",
    ]

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
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for idx, ax in enumerate(axes):
        plot_monatomic_expectation_value(result, idx, ax=ax)
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes
