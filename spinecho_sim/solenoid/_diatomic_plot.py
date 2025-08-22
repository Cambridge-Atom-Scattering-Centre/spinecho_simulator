from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.molecule.hamiltonian_dicke import collective_ops_sparse
from spinecho_sim.util import csr_add, get_figure, sparse_apply, sparse_matmul

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

    from spinecho_sim.solenoid._solenoid import (
        StateVectorSolenoidSimulationResult,
    )


def plot_diatomic_expectation_value(
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
        plot_diatomic_expectation_value(result, idx, "I", ax=axes[idx, 0])
        axes[idx, 0].set_title(rf"$\langle I_{['x', 'y', 'z'][idx]} \rangle$")

        # Plot J expectation values in the second column
        plot_diatomic_expectation_value(result, idx, "J", ax=axes[idx, 1])
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


def plot_diatomic_expectation_difference(
    result: StateVectorSolenoidSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    i_ops, j_ops = collective_ops_sparse(i, j)
    labels = [
        r"\langle I_x \rangle - \langle J_x \rangle",
        r"\langle I_y \rangle - \langle J_y \rangle",
        r"\langle I_z \rangle - \langle J_z \rangle",
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
                np.vdot(state, sparse_apply(i_ops[idx], state))
                - np.vdot(state, sparse_apply(j_ops[idx], state))
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


def plot_diatomic_expectation_differences(
    result: StateVectorSolenoidSimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for idx, ax in enumerate(axes):
        plot_diatomic_expectation_difference(result, idx, ax=ax)

    # Set shared x-axis label for the bottom row
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def compute_diatomic_alignment_tensor(
    result: StateVectorSolenoidSimulationResult,
    spin: Literal["I", "J"],
) -> np.ndarray:
    """Compute the traceless, symmetric rank-2 tensor Q_ij for all components."""
    if spin == "I":
        ops, _ = collective_ops_sparse(
            (result.hilbert_space_dims[0] - 1) / 2,
            (result.hilbert_space_dims[1] - 1) / 2,
        )
        s = (result.hilbert_space_dims[0] - 1) / 2
    else:
        _, ops = collective_ops_sparse(
            (result.hilbert_space_dims[0] - 1) / 2,
            (result.hilbert_space_dims[1] - 1) / 2,
        )
        s = (result.hilbert_space_dims[1] - 1) / 2

    state_vectors = (
        result.state_vectors
    )  # Shape: [number of particles, positions, components]

    # Initialize an array to store Q_ij values (3x3 tensor for each position)
    q_ij_values = np.zeros(
        (3, 3, state_vectors.shape[0], state_vectors.shape[1]), dtype=np.float64
    )

    # Iterate over unique components (i <= j)
    for i in range(3):
        for j in range(i, 3):  # Only compute for i <= j
            for particle_idx in range(state_vectors.shape[0]):
                for position_idx in range(state_vectors.shape[1]):
                    state = state_vectors[
                        particle_idx, position_idx, :
                    ]  # Extract the state vector

                    # Compute the expectation value of F_i F_j + F_j F_i
                    operator = csr_add(
                        sparse_matmul(ops[i], ops[j]), sparse_matmul(ops[j], ops[i])
                    )
                    expectation = np.real(np.vdot(state, sparse_apply(operator, state)))

                    # Compute Q_ij
                    q_ij_values[i, j, particle_idx, position_idx] = (
                        0.5 * expectation - (2 / 3) * s * (s + 1) * (1 if i == j else 0)
                    )

    # Use symmetry to fill in the lower triangular elements
    for i in range(3):
        for j in range(i):
            q_ij_values[i, j, :, :] = q_ij_values[j, i, :, :]

    return q_ij_values


def plot_diatomic_alignment_tensor(
    result: StateVectorSolenoidSimulationResult,
    spin: Literal["I", "J"],
) -> tuple[Figure, Axes]:
    """Plot the traceless, symmetric rank-2 tensor Q_ij for all components."""
    q_ij_values = compute_diatomic_alignment_tensor(result, spin)
    positions = result.positions
    if spin == "I":
        s = (result.hilbert_space_dims[0] - 1) / 2
    else:
        s = (result.hilbert_space_dims[1] - 1) / 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            average_q_ij = np.average(q_ij_values[i, j, :, :], axis=0)
            std_q_ij = np.std(q_ij_values[i, j, :, :], axis=0) / np.sqrt(
                q_ij_values.shape[2]
            )

            # Plot the average Q_ij with error bands
            (line,) = ax.plot(
                positions,
                average_q_ij,
                label=rf"$Q_{{{['x', 'y', 'z'][i]},{['x', 'y', 'z'][j]}}}$",
            )
            color = line.get_color()
            ax.fill_between(
                positions,
                average_q_ij - std_q_ij,
                average_q_ij + std_q_ij,
                alpha=0.2,
                color=color,
                label=rf"$Q_{{{['x', 'y', 'z'][i]},{['x', 'y', 'z'][j]}}} \pm 1\sigma$",
            )

            ax.set_ylabel(rf"$Q_{{{['x', 'y', 'z'][i]},{['x', 'y', 'z'][j]}}}$")
            ax.legend(loc="center left")
            ax.set_xlim(positions[0], positions[-1])
            ax.set_ylim(-s, s)

    # Set shared x-axis label for the bottom row
    axes[-1, 1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes
