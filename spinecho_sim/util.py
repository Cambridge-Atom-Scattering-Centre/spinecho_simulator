from __future__ import annotations

import datetime
import warnings
from collections.abc import Callable, Sequence
from functools import reduce, wraps
from itertools import permutations, starmap
from math import factorial
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast, override

import numpy as np
import scipy.integrate  # pyright: ignore[reportMissingTypeStubs]
import scipy.sparse as sp  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray

from spinecho_sim.field import DataFieldRegion, FieldRegion

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import (
        RendererBase,  # Import RendererBase for type annotations
    )
    from matplotlib.figure import Figure, SubFigure

Vec3 = np.ndarray[tuple[Literal[3]], np.dtype[np.floating]]
Array3 = np.ndarray[tuple[int, Literal[3]], np.dtype[np.floating]]


def get_figure(ax: Axes | None = None) -> tuple[Figure | SubFigure, Axes]:
    """Get a figure and axes for plotting."""
    if ax is None:
        return plt.subplots(figsize=(10, 6))
    return ax.figure, ax


Measure = Literal["real", "imag", "abs", "arg"]


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


def _permute_indices(n_stars: int, perm: tuple[int, ...]) -> np.ndarray:
    """Vectorized map: 0..2**N-1  →  permuted index under `perm`."""
    dim = int(
        2**n_stars
    )  # The total number of basis states in the tensor product space.
    index = np.arange(dim)
    powers = (
        2 ** np.arange(n_stars - 1, -1, -1)
    )  # Computes the powers needed to convert a flat index to its multi-index representation.
    digits = (
        index[:, None] // powers
    ) % 2  # Converts each flat index to its multi-index (i.e., each subsystem state).
    new_digits = digits[
        :, perm
    ]  # Rearranges the digits according to the desired permutation.
    return (new_digits * powers).sum(
        1
    )  # Converts the permuted multi-indices back to flat indices.


def _permutation_matrix(n_stars: int, perm: tuple[int, ...]) -> sp.csr_matrix:
    """Sparse unitary that permutes tensor factors."""
    dim = int(2**n_stars)
    rows = _permute_indices(n_stars, perm)
    cols = np.arange(dim)
    data = np.ones(dim)
    return sp.csr_matrix(sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)))


def csr_add(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    """Typed sp.csr_matrix + sp.csr_matrix → sp.csr_matrix."""
    return (a + b).tocsr()  # type: ignore[operator]


def csr_subtract(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    """Typed sp.csr_matrix - sp.csr_matrix → sp.csr_matrix."""
    return (a - b).tocsr()  # type: ignore[operator]


def csr_scale(a: sp.csr_matrix, scale: complex) -> sp.csr_matrix:
    """Typed sp.csr_matrix * float → sp.csr_matrix."""
    out = a.copy()  # pyright: ignore[reportUnknownVariableType]
    out.data = out.data.astype(complex)  # Ensure the data is complex
    out.data *= scale  # Scale the data by the complex scalar
    return out  # pyright: ignore[reportUnknownVariableType]


def csr_hermitian(a: sp.csr_matrix) -> sp.csr_matrix:
    """Typed sp.csr_matrix† → sp.csr_matrix."""
    return a.transpose().conj().tocsr()  # type: ignore[operator]


def csr_diags(
    diagonals: np.ndarray | list[np.ndarray] | list[float],
    offsets: int = 0,
    shape: tuple[int, int] | None = None,
    dtype: np.dtype | None = None,
) -> sp.csr_matrix:
    """Typed version of sp.diags that guarantees a csr_matrix."""
    m = sp.diags(diagonals, offsets, shape=shape, format="csr", dtype=dtype)
    return cast("sp.csr_matrix", m)


def csr_eye(n: int, dtype: type = complex) -> sp.csr_matrix:
    """Typed identity matrix in sp.csr_matrix format."""
    return cast("sp.csr_matrix", sp.eye(n, dtype=dtype, format="csr"))


def csr_kron(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    """Typed Kronecker product returning sp.csr_matrix."""
    return cast("sp.csr_matrix", sp.kron(a, b, format="csr"))


def symmetrize(n_stars: int) -> sp.csr_matrix:
    """(2^N x 2^N) projector onto the totally-symmetric subspace."""
    projector = sp.csr_matrix((2**n_stars, 2**n_stars), dtype=np.float64)
    for perm in permutations(range(n_stars)):
        projector = csr_add(projector, _permutation_matrix(n_stars, perm))
    projector /= factorial(n_stars)
    return projector  # idempotent, Hermitian


def _spinor(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)],
        dtype=complex,
    )


def product_state(stars: np.ndarray) -> np.ndarray:
    """Create a product state from a list of majorana stars."""
    vectors = list(starmap(_spinor, stars))
    return reduce(np.kron, vectors)


def kronecker_n(operator_list: list[sp.csr_matrix]) -> sp.csr_matrix:
    """Compute the Kronecker product of a list of sparse matrices."""
    return reduce(sp.kron, operator_list)  # type: ignore[return-value]


def csr_to_array(sparse_matrix: sp.csr_matrix) -> np.ndarray:
    """Convert a sparse matrix to a dense numpy array."""
    return sparse_matrix.toarray()  # type: ignore[return-value]


def _normalized_magnitude(arr: np.ndarray) -> np.ndarray:
    """Get the magnitude normalized to [0, 1]."""
    magnitude = np.abs(arr)
    if magnitude.max() > 0:
        return magnitude / magnitude.max()
    return magnitude


def plot_complex_heatmap(arr: np.ndarray) -> tuple[Figure, Axes]:
    """Plot a complex-valued array as a heatmap with phase and magnitude."""
    magnitude = _normalized_magnitude(arr)
    phase = np.angle(arr)

    # Map phase to RGB using hsv colormap
    cmap = plt.get_cmap("hsv")
    norm = Normalize(-np.pi, np.pi)
    rgb = cmap(norm(phase))[..., :3]  # shape (..., 3)

    # Create RGBA image: set alpha to normalized magnitude
    rgba = np.zeros((*arr.shape, 4))
    rgba[..., :3] = rgb
    rgba[..., 3] = magnitude

    fig, ax = plt.subplots()
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Attach colorbar for phase using ScalarMappable, and specify the correct axes
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Phase (radians)")

    return fig, ax


def sparse_matmul(a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
    """Matrix multiplication for two sparse matrices, returning a sparse matrix."""
    return a @ b  # type: ignore[return-value]


def sparse_apply(a: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication for a sparse matrix and a dense vector, returning a dense vector."""
    return a @ b  # type: ignore[return-value]


RHS = Callable[
    [float, NDArray[np.float64 | np.complex128]], NDArray[np.float64 | np.complex128]
]


class SolveIVPResult(Protocol):
    """Result of an ODE solve."""

    # the attributes you read in your code
    t: NDArray[np.float64]
    y: NDArray[np.float64 | np.complex128]
    status: int
    message: str
    success: bool


class SolveIVPOptions(TypedDict, total=False):
    """Options for the ODE solver."""

    atol: float
    rtol: float
    max_step: float
    vectorized: bool


def solve_ivp_typed(  # noqa: PLR0913
    fun: RHS,
    t_span: tuple[float, float],
    y0: NDArray[np.float64 | np.complex128],
    *,
    t_eval: NDArray[np.float64] | None = None,
    method: Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"] = "RK45",
    vectorized: bool = False,
    rtol: float = 1e-3,
    **kwargs: SolveIVPOptions,
) -> SolveIVPResult:
    """Typed wrapper around scipy.integrate.solve_ivp."""
    res = scipy.integrate.solve_ivp(  # pyright: ignore[reportUnknownVariableType]
        fun,
        t_span,
        y0,
        t_eval=t_eval,
        method=method,
        vectorized=vectorized,
        rtol=rtol,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )
    return cast("SolveIVPResult", res)


def verify_hermitian(matrix: sp.csr_matrix) -> bool:
    """Check if a sparse matrix is Hermitian."""
    return (matrix != matrix.getH()).nnz == 0  # pyright: ignore[reportUnknownVariableType]


def check_normalization(psi: np.ndarray, tolerance: float = 1e-8) -> None:
    """Check if the state vector is normalized and issue a warning if not."""
    norm = np.linalg.norm(psi)
    if not np.isclose(norm, 1.0, atol=tolerance):
        warnings.warn(
            f"State vector is not normalized: norm = {norm}", UserWarning, stacklevel=2
        )


class Arrow3D(FancyArrowPatch):
    """Custom class for 3D arrows."""

    def __init__(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        zs: Sequence[float],
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        start_position: tuple[float, float] = (xs[0], ys[0])
        end_position: tuple[float, float] = (xs[1], ys[1])
        super().__init__(*args, posA=start_position, posB=end_position, **kwargs)
        self._verts3d = xs, ys, zs

    @override
    def draw(self, renderer: RendererBase) -> None:
        xs, ys, zs = self._verts3d
        x_proj, y_proj, _ = proj3d.proj_transform(xs, ys, zs, self.axes.M)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportUnknownArgumentType, reportUnknownVariableType]
        self.set_positions((x_proj[0], y_proj[0]), (x_proj[1], y_proj[1]))  # pyright: ignore[reportUnknownArgumentType]
        super().draw(renderer)

    def do_3d_projection(self) -> float:
        """Handle 3D projection for the arrow."""
        xs, ys, zs = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)  # pyright: ignore[reportUnknownVariableType, reportOptionalMemberAccess, reportAttributeAccessIssue, reportUnknownArgumentType]
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))  # pyright: ignore[reportUnknownArgumentType]
        return min(zs)  # pyright: ignore[reportUnknownArgumentType] # Return the minimum z-value for depth sorting


def make_linear_bz_data(  # noqa: PLR0913
    z0: float,
    z1: float,
    bz0: float,
    bz1: float,
    *,
    x_half: float = 0.1,
    y_half: float = 0.1,
    nx: int = 5,
    ny: int = 5,
    nz: int = 6,
) -> DataFieldRegion:
    """Create a DataFieldRegion where Bz varies linearly from bz0 at z0 to bz1 at z1."""
    x = np.linspace(-x_half, x_half, nx)
    y = np.linspace(-y_half, y_half, ny)
    z = np.linspace(z0, z1, nz)
    field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for ix, _xx in enumerate(x):
        for iy, _yy in enumerate(y):
            for iz, zz in enumerate(z):
                t = 0.0 if z1 == z0 else (zz - z0) / (z1 - z0)
                bz = (1.0 - t) * bz0 + t * bz1
                field[ix, iy, iz, :] = [0.0, 0.0, bz]
    return FieldRegion.from_data(x_vals=x, y_vals=y, z_vals=z, field_data=field)


def make_bx_blob(  # noqa: PLR0913
    *,
    x_half: float = 0.1,
    y_half: float = 0.1,
    z0: float = 0.0,
    z1: float = 1.0,
    nx: int = 21,
    ny: int = 21,
    nz: int = 11,
    r_scale: float = 0.1,
    z_center: float | None = None,
    z_width: float = 0.2,
    amplitude: float = 0.1,
) -> DataFieldRegion:
    """Create a localized Bx 'blob' with Gaussian radial and axial profiles."""
    if z_center is None:
        z_center = 0.5 * (z0 + z1)
    x = np.linspace(-x_half, x_half, nx)
    y = np.linspace(-y_half, y_half, ny)
    z = np.linspace(z0, z1, nz)
    vals = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            for iz, zz in enumerate(z):
                bx = (
                    amplitude
                    * np.exp(-((xx**2 + yy**2) / (r_scale**2)))
                    * np.exp(-((zz - z_center) ** 2) / (z_width**2))
                )
                vals[ix, iy, iz, 0] = bx
    return FieldRegion.from_data(x_vals=x, y_vals=y, z_vals=z, field_data=vals)
