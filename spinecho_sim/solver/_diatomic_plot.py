from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt

from spinecho_sim.molecule.hamiltonian_dicke import build_collective_operators
from spinecho_sim.util import csr_add, get_figure, sparse_apply, sparse_matmul

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

    from spinecho_sim.solver._solver import (
        StateVectorSimulationResult,
    )


def plot_diatomic_expectation_value(
    result: StateVectorSimulationResult,
    idx: int,
    spin: Literal["I", "J"],
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    if spin == "I":
        ops, _ = build_collective_operators(i, j)
    else:
        _, ops = build_collective_operators(i, j)

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
    result: StateVectorSimulationResult,
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
    result: StateVectorSimulationResult,
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
    result: StateVectorSimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    i_ops, j_ops = build_collective_operators(i, j)
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
    result: StateVectorSimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for idx, ax in enumerate(axes):
        plot_diatomic_expectation_difference(result, idx, ax=ax)

    # Set shared x-axis label for the bottom row
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def compute_diatomic_alignment_tensor(
    result: StateVectorSimulationResult,
    spin: Literal["I", "J"],
) -> np.ndarray:
    """Compute the traceless, symmetric rank-2 tensor Q_ij for all components."""
    if spin == "I":
        ops, _ = build_collective_operators(
            (result.hilbert_space_dims[0] - 1) / 2,
            (result.hilbert_space_dims[1] - 1) / 2,
        )
        s = (result.hilbert_space_dims[0] - 1) / 2
    else:
        _, ops = build_collective_operators(
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
                        0.5 * expectation - (1 / 3) * s * (s + 1) * (1 if i == j else 0)
                    )

    # Use symmetry to fill in the lower triangular elements
    for i in range(3):
        for j in range(i):
            q_ij_values[i, j, :, :] = q_ij_values[j, i, :, :]

    return q_ij_values


def plot_diatomic_alignment_tensor(
    result: StateVectorSimulationResult,
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


def plot_diatomic_alignment_diagnostics(  # noqa: PLR0914
    result: StateVectorSimulationResult,
    spin: Literal["I", "J"],
) -> tuple[Figure, Axes]:
    q_tensor = compute_diatomic_alignment_tensor(
        result, spin
    )  # shape: [3, 3, number of particles, positions]
    assert q_tensor.shape[0:2] == (3, 3)
    number_particles, number_positions = q_tensor.shape[2], q_tensor.shape[3]

    positions = result.positions
    if spin == "I":
        f = (result.hilbert_space_dims[0] - 1) / 2
    else:
        f = (result.hilbert_space_dims[1] - 1) / 2
    normalized_q_tensor = q_tensor / (f * (f + 1))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    # --- batch eigen-decomposition over (particle, position) --------------------
    # reshape to (Np*Nz, 3, 3)
    q_reshaped = normalized_q_tensor.transpose(2, 3, 0, 1).reshape(
        number_particles * number_positions, 3, 3
    )  # (T,3,3) with T= Np*Nz

    # eigh returns ascending eigenvalues; eigenvectors as columns
    eigenvalues_reshaped, _ = np.linalg.eigh(q_reshaped)  # w: (T,3), V: (T,3,3)

    # reorder to descending eigenvalues and matching eigenvectors
    eigenvalues_reshaped = eigenvalues_reshaped[:, ::-1]  # (T,3)

    # reshape back to (3, Np, Nz)
    eigenvalues = eigenvalues_reshaped.reshape(
        number_particles, number_positions, 3
    ).transpose(2, 0, 1)  # (3, Np, Nz)

    # --- uniaxial S and biaxiality η per particle/position ---------------------
    # largest eigenvalue is lam[0]
    s = 1.5 * eigenvalues[0, :, :]  # (Np, Nz)
    eta = (eigenvalues[1, :, :] - eigenvalues[2, :, :]) / eigenvalues[0, :, :]

    # --- scalar invariants q and beta per particle/position --------------------
    trace_q_tensor_squared = np.einsum("tij,tij->t", q_reshaped, q_reshaped)  # (T,)
    trace_q_tensor_cubed = np.einsum(
        "tij,tjk,tki->t", q_reshaped, q_reshaped, q_reshaped
    )  # (T,)
    q = np.sqrt(1.5 * trace_q_tensor_squared).reshape(
        number_particles, number_positions
    )  # (Np, Nz)
    beta = (
        1.0 - 6.0 * (trace_q_tensor_cubed**2) / (trace_q_tensor_squared**3 + 0.0)
    ).reshape(number_particles, number_positions)

    # --- particle averages at each position ------------------------------------
    s_mean = np.mean(s, axis=0)  # (Nz,)
    s_std = np.std(s, axis=0) / np.sqrt(number_particles)  # (Nz,)
    eta_mean = np.mean(eta, axis=0)  # (Nz,)
    eta_std = np.std(eta, axis=0) / np.sqrt(number_particles)  # (Nz,)
    q_mean = np.mean(q, axis=0)  # (Nz,)
    q_std = np.std(q, axis=0) / np.sqrt(number_particles)  # (Nz,)
    beta_mean = np.mean(beta, axis=0)  # (Nz,)
    beta_std = np.std(beta, axis=0) / np.sqrt(number_particles)  # (Nz,)

    (line,) = axes[0, 0].plot(
        positions,
        s_mean,
        label=r"$S$",
    )
    color = line.get_color()
    axes[0, 0].fill_between(
        positions,
        s_mean - s_std,
        s_mean + s_std,
        alpha=0.2,
        color=color,
        label=r"$S \pm 1\sigma$",
    )
    axes[0, 0].legend(loc="center left")
    axes[0, 0].set_ylim(-0.5, 1)

    (line,) = axes[1, 0].plot(
        positions,
        eta_mean,
        label=r"$\eta$",
    )
    color = line.get_color()
    axes[1, 0].fill_between(
        positions,
        eta_mean - eta_std,
        eta_mean + eta_std,
        alpha=0.2,
        color=color,
        label=r"$\eta \pm 1\sigma$",
    )
    axes[1, 0].legend(loc="center left")

    (line,) = axes[0, 1].plot(
        positions,
        q_mean,
        label=r"$q$",
    )
    color = line.get_color()
    axes[0, 1].fill_between(
        positions,
        q_mean - q_std,
        q_mean + q_std,
        alpha=0.2,
        color=color,
        label=r"$q \pm 1\sigma$",
    )
    axes[0, 1].legend(loc="center left")
    axes[0, 1].set_ylim(0, 1)

    (line,) = axes[1, 1].plot(
        positions,
        beta_mean,
        label=r"$\beta$",
    )
    color = line.get_color()
    axes[1, 1].fill_between(
        positions,
        beta_mean - beta_std,
        beta_mean + beta_std,
        alpha=0.2,
        color=color,
        label=r"$\beta \pm 1\sigma$",
    )
    axes[1, 1].legend(loc="center left")
    axes[1, 1].set_ylim(0, 1)

    # Set shared x-axis label for the bottom row
    axes[-1, 0].set_xlabel(r"Distance $z$ along Solenoid Axis")
    axes[-1, 1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes
