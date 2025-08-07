from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp  # type: ignore[import]
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

sigma_x = sp.csr_matrix([[0, 1], [1, 0]])
sigma_y = sp.csr_matrix([[0, -1j], [1j, 0]])
sigma_z = sp.csr_matrix([[1, 0], [0, -1]])
identity = sp.csr_matrix([[1, 0], [0, 1]])


def kronecker_n(ops: list[sp.csr_matrix]) -> sp.csr_matrix:
    """Compute the Kronecker product of a list of sparse matrices."""
    return reduce(sp.kron, ops)  # type: ignore[return-value]


def zeeman_block(
    n_i: int,
    n_j: int,
    coefficient: float,
    b_vec: tuple[float, float, float],
    constellation: Literal["I", "J"],
) -> sp.csr_matrix:
    if constellation == "I":
        n = n_i
        start = 0
    else:
        n = n_j
        start = n_i

    hamiltonian = sp.csr_matrix((2 ** (n_i + n_j),) * 2, dtype=complex)
    for k in range(n):
        operators = [identity] * (n_i + n_j)
        for component, operator in zip(b_vec, (sigma_x, sigma_y, sigma_z), strict=True):
            operators[start + k] = operator
            hamiltonian += coefficient * component * kronecker_n(operators)
            operators[start + k] = identity
    return hamiltonian


def zeeman_hamiltonian(
    n_i: int,
    n_j: int,
    coefficients: tuple[float, float],
    b_vec: tuple[float, float, float],
) -> sp.csr_matrix:
    """Construct the Zeeman Hamiltonian for two spin systems."""
    hamiltonian = sp.csr_matrix((2 ** (n_i + n_j),) * 2, dtype=complex)
    hamiltonian += zeeman_block(n_i, n_j, coefficients[0], b_vec, "I")
    hamiltonian += zeeman_block(n_i, n_j, coefficients[1], b_vec, "J")
    return hamiltonian


def plot_complex_heatmap(arr: np.ndarray) -> tuple[Figure, Axes]:
    magnitude = np.abs(arr)
    phase = np.angle(arr)

    # Normalize magnitude to [0, 1] for alpha
    mag_norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude

    # Map phase to RGB using hsv colormap
    cmap = plt.get_cmap("hsv")
    norm = Normalize(-np.pi, np.pi)
    rgb = cmap(norm(phase))[..., :3]  # shape (..., 3)

    # Create RGBA image: set alpha to normalized magnitude
    rgba = np.zeros((*arr.shape, 4))
    rgba[..., :3] = rgb
    rgba[..., 3] = mag_norm

    fig, ax = plt.subplots()
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Attach colorbar for phase using ScalarMappable, and specify the correct axes
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Phase (radians)")

    return fig, ax


result = zeeman_block(2, 2, 1.0, (0.1, 0.1, 1.0), "I").toarray()
print(result)
fig, ax = plot_complex_heatmap(result)

result = zeeman_block(2, 2, 1.0, (0.1, 0.1, 1.0), "J").toarray()
print(result)
fig, ax = plot_complex_heatmap(result)

result = zeeman_hamiltonian(2, 2, (1.0, 1.0), (0.1, 0.1, 1.0)).toarray()
print(result)
fig, ax = plot_complex_heatmap(result)

plt.show()
