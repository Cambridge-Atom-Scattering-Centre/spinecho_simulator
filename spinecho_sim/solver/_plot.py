from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from tqdm import tqdm

from spinecho_sim.molecule.hamiltonian_dicke import build_collective_operators
from spinecho_sim.state import get_expectation_values
from spinecho_sim.util import (
    Arrow3D,
    Measure,
    get_figure,
    get_measure_label,
    measure_data,
    sparse_apply,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    from spinecho_sim.solver._solver import (
        SimulationResult,
        StateVectorSimulationResult,
    )
    from spinecho_sim.state._trajectory import (
        Trajectory,
    )


def plot_monatomic_spin_state(
    result: SimulationResult,
    idx: int,
    *,
    measure: Measure = "abs",
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spin.momentum_states[idx, :, :]
    state_measure = measure_data(states, measure)

    average_state_measure = np.average(state_measure, axis=0)

    n_stars = result.spin.n_stars
    s = n_stars / 2
    ms_values = np.linspace(s, -s, n_stars + 1, endpoint=True)
    ms_labels = [
        rf"$|m_S={int(m)} \rangle$"
        if m.is_integer()
        else rf"$|m_S={2 * m:.0f}/2 \rangle$"
        for m in ms_values
    ]

    # Plot phase
    (measure_line,) = ax.plot(positions, average_state_measure, label="Mean")
    color_measure = measure_line.get_color()
    ax.plot(
        positions,
        np.swapaxes(state_measure, 0, 1).reshape(positions.size, -1),
        alpha=0.1,
        color=color_measure,
    )

    # Standard error of the mean for phase
    std_states_measure = np.std(state_measure, axis=0) / np.sqrt(len(states))
    ax.fill_between(
        positions,
        (average_state_measure - std_states_measure).ravel(),
        (average_state_measure + std_states_measure).ravel(),
        alpha=0.2,
        linestyle=":",
        color=color_measure,
        label=r"Mean $\pm 1\sigma$",
    )

    ax.set_ylabel(f"{ms_labels[idx]} {get_measure_label(measure)}")
    ax.legend(loc="lower right")
    ax.set_xlim(positions[0], positions[-1])

    return fig, ax


def plot_monatomic_state_intensity(
    result: SimulationResult, idx: int, *, ax: Axes | None = None
) -> tuple[Figure | SubFigure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    positions = result.positions
    states = result.spin.momentum_states[idx]
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


def plot_monatomic_spin_states(result: SimulationResult) -> tuple[Figure, Axes]:
    n_stars = result.spin.n_stars
    fig, axes = plt.subplots(n_stars + 1, 2, figsize=(10, 6), sharex=True)

    for idx, (ax_abs, ax_arg) in enumerate(axes):
        plot_monatomic_spin_state(result, idx, measure="abs", ax=ax_abs)
        plot_monatomic_spin_state(result, idx, measure="arg", ax=ax_arg)
    for ax in axes[-1]:
        ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_monatomic_expectation_value(
    result: SimulationResult,
    idx: int,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)[idx, :]

    average_state_measure = np.average(expectation_values, axis=0)
    labels = [
        r"\langle S_x \rangle",
        r"\langle S_y \rangle",
        r"\langle S_z \rangle",
    ]

    (measure_line,) = ax.plot(positions, average_state_measure)
    measure_line.set_label(rf"$\overline{{{labels[idx]}}} / \hbar$")
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


def plot_monatomic_expectation_values(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for idx, ax in enumerate(axes):
        plot_monatomic_expectation_value(result, idx, ax=ax)
    axes[-1].set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, axes


def plot_monatomic_expectation_phi(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)

    wrapped_phi = np.arctan2(
        expectation_values[1, :], expectation_values[0, :]
    )  # atan2(y, x) gives the angle in radians
    phi = np.unwrap(wrapped_phi, axis=1) / np.pi  # Unwrap and normalize to [0, 2π)

    average_phi = np.average(phi, axis=0)

    (average_line,) = ax.plot(positions, average_phi)
    average_line.set_label(r"$\overline{\langle \phi \rangle}$")
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
    ax.legend(loc="upper right")
    ax.set_ylabel(r"$\langle \phi \rangle$ Azimuthal Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_monatomic_expectation_theta(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, Axes]:
    fig, ax = get_figure(ax)

    positions = result.positions
    expectation_values = get_expectation_values(result.spin)

    wrapped_theta = np.arctan2(
        np.sqrt(expectation_values[0, :] ** 2 + expectation_values[1, :] ** 2),
        expectation_values[2, :],
    )
    theta = np.unwrap(wrapped_theta, axis=1) / np.pi

    average_theta = np.average(theta, axis=0)

    (average_line,) = ax.plot(positions, average_theta)
    average_line.set_label(r"$\overline{\langle \theta \rangle}$")
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
    ax.legend(loc="upper left")
    ax.set_ylabel(r"$\langle \theta \rangle$ Polar Angle (radians/$\pi$)")
    ax.set_xlim(positions[0], positions[-1])
    return fig, ax


def plot_monatomic_expectation_angles(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_monatomic_expectation_theta(result, ax=ax)
    plot_monatomic_expectation_phi(result, ax=ax.twinx())
    ax.set_xlabel(r"Distance $z$ along Solenoid Axis")
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title

    return fig, ax


def _plot_monatomic_expectation_trajectory(
    trajectory: Trajectory,
    fig: Figure,
    ax: Axes3D,
) -> tuple[Figure, Axes3D, Line2D]:
    expectations = get_expectation_values(trajectory.spin)

    # Plot the trajectory as a 3D curve
    (line,) = ax.plot(expectations[0, :], expectations[1, :], expectations[2, :])
    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    return fig, ax, line


def plot_monatomic_expectation_trajectories(
    result: SimulationResult,
) -> tuple[Figure, Axes3D]:
    fig = plt.figure(figsize=(8, 8))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))

    expectations = get_expectation_values(result.trajectories.spin)
    # Average over samples (axis=1), shape: (3, n_positions)
    avg_expectations = np.average(expectations, axis=1)

    # Unpack components
    x = avg_expectations[0, :]
    y = avg_expectations[1, :]
    z = avg_expectations[2, :]

    # Plot the trajectory as a 3D curve
    (average_line,) = ax.plot(x, y, z)
    average_line.set_label(r"$\overline{\langle \mathbf{S} \rangle}$")
    color = average_line.get_color()
    for trajectory in result.trajectories:
        _, _, line = _plot_monatomic_expectation_trajectory(trajectory, fig, ax)
        line.set_alpha(0.1)
        line.set_color(color)

    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # 0 - 0.95 for axes, 0.05 for title
    return fig, ax


def _calculate_expectation_values(
    result: StateVectorSimulationResult,
    spin: Literal["I", "J"],
) -> np.ndarray:
    """Calculate the expectation values for the given spin."""
    i = (result.hilbert_space_dims[0] - 1) / 2
    j = (result.hilbert_space_dims[1] - 1) / 2

    if spin == "I":
        ops, _ = build_collective_operators(i, j)
    else:
        _, ops = build_collective_operators(i, j)

    positions = result.positions
    state_vectors = (
        result.state_vectors
    )  # Shape: [number of particles, positions, components]

    # Calculate expectation values for each position
    expectation_values = np.zeros(
        (state_vectors.shape[0], positions.size, 3), dtype=np.float64
    )
    for particle_idx in range(state_vectors.shape[0]):
        for position_idx in range(positions.size):
            state = state_vectors[
                particle_idx, position_idx, :
            ]  # Extract state vector for this particle at this position
            expectation_values[particle_idx, position_idx, 0] = np.real(
                np.vdot(state, sparse_apply(ops[0], state))
            )
            expectation_values[particle_idx, position_idx, 1] = np.real(
                np.vdot(state, sparse_apply(ops[1], state))
            )
            expectation_values[particle_idx, position_idx, 2] = np.real(
                np.vdot(state, sparse_apply(ops[2], state))
            )

    return expectation_values


def _setup_figure_and_axes(spin: float) -> tuple[Figure, Axes3D]:
    fig = plt.figure(figsize=(10, 10))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-spin, spin])
    ax.set_ylim([-spin, spin])
    ax.set_zlim([-spin, spin])
    ax.set_xlabel(r"$\langle S_x \rangle$")
    ax.set_ylabel(r"$\langle S_y \rangle$")
    ax.set_zlabel(r"$\langle S_z \rangle$")
    ax.set_title(r"$3$D Expectation Vector $\langle \mathbf{S} \rangle$ Animation")
    return fig, ax


def _add_coordinate_planes(ax: Axes3D, spin: float) -> None:
    x, y = np.meshgrid(np.array([-spin, spin]), np.array([-spin, spin]))
    ax.plot_surface(x, y, 0 * x, alpha=0.07, color="C0")  # z=0 (xy)
    ax.plot_surface(x, 0 * x, y, alpha=0.07, color="C1")  # y=0 (xz)
    ax.plot_surface(0 * x, x, y, alpha=0.07, color="C2")  # x=0 (yz)


def _add_atom(ax: Axes3D, spin: float, r_frac: float = 0.05) -> None:
    r = r_frac * spin
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    xs = r * np.cos(u) * np.sin(v)
    ys = r * np.sin(u) * np.sin(v)
    zs = r * np.cos(v)
    ax.plot_surface(
        xs,
        ys,
        zs,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=0.55,
        color="#9aa0a6",
        zorder=0,
    )


def _make_arrow(
    label: str, color: str = "k", lw: float = 1.5, scale: float = 20
) -> Artist:
    return Arrow3D(
        [0, 0],
        [0, 0],
        [0, 0],
        mutation_scale=scale,  # pyright: ignore[reportArgumentType]
        lw=lw,  # pyright: ignore[reportArgumentType]
        color=color,  # pyright: ignore[reportArgumentType]
        label=label,  # pyright: ignore[reportArgumentType]
    )


def _animate_vectors_core(  # noqa: C901, PLR0913, PLR0914, PLR0915
    expectations_by_name: Mapping[str, np.ndarray],  # each: shape (3, n)
    *,
    box_limit: float,
    labels_colors: Mapping[str, str],  # name -> main arrow color
    title: str,
    axis_labels: Sequence[str] = (
        r"$\langle S_x \rangle$",
        r"$\langle S_y \rangle$",
        r"$\langle S_z \rangle$",
    ),
    show_planes: bool = True,
    show_atom: bool = False,
    atom_r_frac: float = 0.05,
    # --- trails ---
    with_trail: bool = False,
    trail_length: int = 40,
    trail_alpha: float = 0.4,
    trail_lw: float = 1.0,
    trail_colors: Mapping[str, str] | None = None,  # override per vector if desired
    # --- drops ---
    with_drops: bool = False,
    drop_axis_colors: tuple[str, str, str] = ("C2", "C1", "C0"),  # (x,y,z)
    drop_lw: float = 0.5,
    drop_alpha: float = 0.8,
    drop_scale: float = 10.0,
    drop_show_in_legend: bool = False,  # set True to show component labels (first vector only)
    drop_label_templates: tuple[str, str, str] = (
        r"$\langle S_x \rangle$",
        r"$\langle S_y \rangle$",
        r"$\langle S_z \rangle$",
    ),
    # --- HUD ---
    hud_formatter: Callable[[dict[str, tuple[float, float, float]]], str] | None = None,
    frames: int | None = None,
) -> FuncAnimation:
    """Create a generic 3D expectation-vector animator for mono/diatomic variants."""
    fig, ax = _setup_figure_and_axes(box_limit)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)

    if show_planes:
        _add_coordinate_planes(ax, box_limit)
    if show_atom:
        _add_atom(ax, box_limit, r_frac=atom_r_frac)

    # build arrows
    arrows: dict[str, Artist] = {}
    for name, color in labels_colors.items():
        arr = _make_arrow(label=name, color=color, lw=2, scale=20)
        ax.add_artist(arr)
        arrows[name] = arr

    drops: dict[str, dict[str, Artist]] = {}
    if with_drops:
        first = True
        for name in expectations_by_name:
            labels = (
                drop_label_templates
                if (drop_show_in_legend and first)
                else ("_nolegend_",) * 3
            )
            dx = _make_arrow(
                label=labels[0], color=drop_axis_colors[0], lw=drop_lw, scale=drop_scale
            )
            dy = _make_arrow(
                label=labels[1], color=drop_axis_colors[1], lw=drop_lw, scale=drop_scale
            )
            dz = _make_arrow(
                label=labels[2], color=drop_axis_colors[2], lw=drop_lw, scale=drop_scale
            )
            for d in (dx, dy, dz):
                cast("Any", d).set_alpha(drop_alpha)
                ax.add_artist(d)
            drops[name] = {"x": dx, "y": dy, "z": dz}
            first = (
                False  # only the first vector contributes labels (avoids duplicates)
            )

    # --- NEW: one trail per vector ---
    trails: dict[str, Line2D] = {}
    if with_trail:
        for name in expectations_by_name:
            color = (
                trail_colors[name]
                if (trail_colors and name in trail_colors)
                else labels_colors.get(name, "k")
            )
            (line,) = ax.plot([], [], [], color=color, alpha=trail_alpha, lw=trail_lw)
            trails[name] = line

    # HUD text (optional)
    txt: Text | None = None
    if hud_formatter is not None:
        txt = ax.text2D(0.70, 0.98, "", ha="left", va="top", transform=ax.transAxes)

    ax.legend(loc="upper left")

    # figure out frames
    n_frames = (
        frames
        if frames is not None
        else min(v.shape[1] for v in expectations_by_name.values())
    )

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=n_frames, desc="Animating Frames")

    # update function
    def _update(i: int) -> tuple[Artist, ...]:
        current_vals: dict[str, tuple[float, float, float]] = {}
        for name, vectors in expectations_by_name.items():
            xi, yi, zi = vectors[0, i], vectors[1, i], vectors[2, i]
            arrows[name]._verts3d = [0, xi], [0, yi], [0, zi]  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]  # noqa: SLF001
            current_vals[name] = (float(xi), float(yi), float(zi))

            if with_drops:
                d = drops[name]
                d["x"]._verts3d = ([0, xi], [0, 0], [0, 0])  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
                d["y"]._verts3d = ([0, 0], [0, yi], [0, 0])  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
                d["z"]._verts3d = ([0, 0], [0, 0], [0, zi])  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001

        # --- update every trail ---
        if with_trail:
            j0 = max(0, i - trail_length)
            for name, line in trails.items():
                arr = expectations_by_name[name]
                line.set_data_3d(  # type: ignore[attr-defined]
                    arr[0, j0 : i + 1], arr[1, j0 : i + 1], arr[2, j0 : i + 1]
                )

        if txt is not None and hud_formatter is not None:
            txt.set_text(hud_formatter(current_vals))

        # Update the progress bar
        progress_bar.update(1)

        # return artists (helps even with blit=False)
        ret: list[Artist] = []
        ret.extend(arrows.values())
        if with_drops:
            for d in drops.values():
                ret.extend(d.values())
        if with_trail:
            ret.extend(tr for tr in trails.values())
        if txt is not None:
            ret.append(txt)
        return tuple(ret)

    # Close the progress bar when the animation is complete
    def _close_progress_bar(progress_bar: tqdm[NoReturn]) -> Callable[[], None]:
        """Return a function to close the progress bar."""

        def close() -> None:
            progress_bar.close()

        return close

    # Create the animation
    anim = FuncAnimation(fig, _update, frames=n_frames, interval=50, blit=False)

    # Attach the progress bar close function to the animation's event source
    anim.event_source.add_callback(_close_progress_bar(progress_bar))
    return anim


def animate_monatomic_mean_expectation_vectors(
    result: SimulationResult,
) -> FuncAnimation:
    positions = result.positions
    expectations = get_expectation_values(result.trajectories.spin)
    avg_expectations = np.average(expectations, axis=1)  # (3, n)
    spin = result.trajectories.spin.n_stars / 2

    s_label = "S"  # machine key
    return _animate_vectors_core(
        expectations_by_name={s_label: avg_expectations},
        box_limit=spin,
        labels_colors={s_label: "k"},
        title=r"$3$D Expectation Vector $\langle \mathbf{S} \rangle$ Animation",
        axis_labels=(
            r"$\langle S_x \rangle$",
            r"$\langle S_y \rangle$",
            r"$\langle S_z \rangle$",
        ),
        show_planes=True,
        show_atom=True,
        atom_r_frac=0.05,
        with_trail=True,
        with_drops=True,  # <-- enable projections
        drop_show_in_legend=True,  # show ⟨Sx⟩, ⟨Sy⟩, ⟨Sz⟩ once
        hud_formatter=lambda vals: rf"$\langle \mathbf{{S}} \rangle = ({vals[s_label][0]:+.3f}, {vals[s_label][1]:+.3f}, {vals[s_label][2]:+.3f})$",
        frames=positions.size,
    )


def animate_diatomic_mean_expectation_vectors(
    result: StateVectorSimulationResult,
) -> FuncAnimation:
    positions = result.positions

    # (3, n) each
    expectation_values_i = np.average(
        _calculate_expectation_values(result, spin="I"), axis=0
    ).T
    expectation_values_j = np.average(
        _calculate_expectation_values(result, spin="J"), axis=0
    ).T

    def hud(vals: dict[str, tuple[float, float, float]]) -> str:
        xi, yi, zi = vals["I"][0], vals["I"][1], vals["I"][2]
        xj, yj, zj = vals["J"][0], vals["J"][1], vals["J"][2]
        return (
            rf"$I=({xi:+.3f},{yi:+.3f},{zi:+.3f})$"
            "\n"
            rf"$J=({xj:+.3f},{yj:+.3f},{zj:+.3f})$"
        )

    return _animate_vectors_core(
        expectations_by_name={
            "I": expectation_values_i,
            "J": expectation_values_j,
        },
        box_limit=min(result.hilbert_space_dims) / 2,
        labels_colors={
            "I": "blue",
            "J": "red",
        },
        title=r"$3$D Expectation Vector Animation (Nuclear $\langle \mathbf{I} \rangle$ and Rotational $\langle \mathbf{J} \rangle$ Angular Momentum)",
        axis_labels=(
            r"$\langle X \rangle$",
            r"$\langle Y \rangle$",
            r"$\langle Z \rangle$",
        ),
        show_planes=True,
        show_atom=False,  # diatomic: disable atom sphere by default
        with_trail=True,  # or True if you want a trail (uses first entry)
        with_drops=True,
        hud_formatter=hud,
        frames=positions.size,
    )
