from __future__ import annotations

import numpy as np
import scipy.sparse as sp  # type: ignore[import]


def spin_operators(
    s: float, *, fmt: str = "csr"
) -> tuple[sp.spmatrix, sp.spmatrix, sp.spmatrix]:
    """Return sparse (Sx,Sy,Sz) for spin S in the {|m⟩} basis, -S≤m≤S."""
    m = np.arange(s, -s - 1, -1)  # descending for conventional order
    sz = sp.diags(m, format=fmt)

    m_up = m[:-1]  # ladder-operator prefactors
    coef_p = np.sqrt(s * (s + 1) - m_up * (m_up - 1))
    s_plus = sp.diags(coef_p, 1, format=fmt)
    s_minus = s_plus.T.conj()

    sx = (s_plus + s_minus) / 2
    sy = (s_plus - s_minus) / (2j)
    return sx, sy, sz


# choose angular momenta -----------------------------------------------------
I, J = 1, 1  # <-- change here for higher states
SxI, SyI, SzI = spin_operators(I)
SxJ, SyJ, SzJ = spin_operators(J)

# embed in the direct-product Hilbert space ( (2I+1)*(2J+1) )
NI, NJ = int(2 * I + 1), int(2 * J + 1)
Ix, Iy, Iz = [sp.kron(op, sp.eye(NJ), format="csr") for op in (SxI, SyI, SzI)]
Jx, Jy, Jz = [sp.kron(sp.eye(NI), op, format="csr") for op in (SxJ, SyJ, SzJ)]
