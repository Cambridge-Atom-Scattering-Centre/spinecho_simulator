from __future__ import annotations

import datetime
from functools import wraps
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


def get_figure(ax: Axes | None = None) -> tuple[Figure | SubFigure, Axes]:
    """Get a figure and axes for plotting."""
    if ax is None:
        return plt.subplots(figsize=(10, 6))
    return ax.figure, ax


Measure = Literal["real", "imag", "abs", "arg"]


def measure_data(arr: np.ndarray, measure: Measure) -> np.ndarray:
    """Get the specified measure of an array."""
    if measure == "real":
        return np.real(arr)
    if measure == "imag":
        return np.imag(arr)
    if measure == "abs":
        return _signed_mag_and_phase(arr)[0]
    if measure == "arg":
        return _signed_mag_and_phase(arr)[1] / np.pi
    return None


def get_measure_label(measure: Measure) -> str:
    """Get the specified measure of an array."""
    if measure == "real":
        return "Real part"
    if measure == "imag":
        return "Imaginary part"
    if measure == "abs":
        return "Magnitude"
    if measure == "arg":
        return r"Phase $/\pi$"
    return None


def _signed_mag_and_phase(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = arr.shape
    n_particles = shape[0]
    rest_shape = shape[1:]

    m_signed = np.empty_like(arr, dtype=np.float64)
    phi_signed = np.empty_like(arr, dtype=np.float64)

    for i in range(n_particles):
        arr_flat = np.asarray(arr[i].ravel(), dtype=np.complex128)
        n = arr_flat.size
        phi = np.unwrap(np.angle(arr_flat))  # raw phase in (-π,π]
        mag = np.abs(arr_flat)
        m_s = mag.copy()
        phi_s = phi.copy()

        for k in range(1, n):
            phase_change = phi_s[k] - phi_s[k - 1]
            # detect a +π-jump
            if phase_change > np.pi / 2:
                m_s[k:] *= -1
                phi_s[k:] -= np.pi
            # detect a -π-jump
            elif phase_change < -np.pi / 2:
                m_s[k:] *= -1
                phi_s[k:] += np.pi

        m_signed[i] = m_s.reshape(rest_shape)
        phi_signed[i] = phi_s.reshape(rest_shape)

    return m_signed, phi_signed


def timed[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """
    Log the time taken for f to run.

    Parameters
    ----------
    f : Callable[P, R]
        The function to time

    Returns
    -------
    Callable[P, R]
        The decorated function
    """

    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        try:
            result = f(*args, **kw)
        finally:
            te = datetime.datetime.now(tz=datetime.UTC)
            print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")  # noqa: T201
        return result

    return wrap  # type: ignore[return-value]
