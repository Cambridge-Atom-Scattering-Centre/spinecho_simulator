from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload, override

import numpy as np

from spinecho_sim.measurement._spin_ladder_operators import transverse_expectation
from spinecho_sim.state import CoherentSpin, Spin


@dataclass(kw_only=True, frozen=True)
class SpinExpectation:
    """Represents the transverse expectation values of spin components for a single particle."""

    jx: float
    jy: float

    @property
    def x(self) -> float:
        """Get the x-coordinate of the displacement."""
        return self.jx

    @property
    def y(self) -> float:
        """Get the y-coordinate of the displacement."""
        return self.jy

    @staticmethod
    def from_spin(spin: np.ndarray[Any, np.dtype[np.complex128]]) -> SpinExpectation:
        """Create a SpinExpectation from a spin state."""
        state = Spin.from_iter([CoherentSpin(theta=s[0], phi=s[1]) for s in spin])
        expectation = transverse_expectation(state.momentum_states)
        return SpinExpectation(
            jx=expectation[0],
            jy=expectation[1],
        )


@dataclass(kw_only=True, frozen=True)
class SpinExpectationList(Sequence[SpinExpectation]):
    """A list of SpinExpectation objects."""

    jx: np.ndarray[Any, np.dtype[np.floating]]
    jy: np.ndarray[Any, np.dtype[np.floating]]

    def __post_init__(self) -> None:
        if self.jx.shape != self.jy.shape:
            msg = "jx and jy arrays must have the same shape."
            raise ValueError(msg)

    @staticmethod
    def from_expectations(
        expectations: Iterable[SpinExpectation],
    ) -> SpinExpectationList:
        """Create a SpinExpectationList from a list of SpinExpectations."""
        expectations = list(expectations)
        return SpinExpectationList(
            jx=np.array([e.jx for e in expectations]),
            jy=np.array([e.jy for e in expectations]),
        )

    @staticmethod
    def from_spins(
        spins: Iterable[np.ndarray[Any, np.dtype[np.complex128]]],
    ) -> SpinExpectationList:
        """Create a SpinExpectationList from a list of spin states."""
        expectations = [SpinExpectation.from_spin(spin) for spin in spins]
        return SpinExpectationList.from_expectations(expectations)

    @override
    def __len__(self) -> int:
        return self.jx.shape[0]

    @overload
    def __getitem__(self, index: int) -> SpinExpectation: ...

    @overload
    def __getitem__(self, index: slice) -> SpinExpectationList: ...

    @override
    def __getitem__(self, index: int | slice) -> SpinExpectation | SpinExpectationList:
        if isinstance(index, slice):
            return SpinExpectationList(jx=self.jx[index], jy=self.jy[index])
        return SpinExpectation(jx=self.jx.item(index), jy=self.jy.item(index))

    @override
    def __iter__(self) -> Iterator[SpinExpectation]:
        for jx, jy in zip(self.jx, self.jy, strict=True):
            yield SpinExpectation(jx=jx, jy=jy)

    @property
    def x(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the S_x expectation value of the particles."""
        return np.array([expectation.jx for expectation in self])

    @property
    def y(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Get the S_y expectation value of the particles."""
        return np.array([expectation.jy for expectation in self])

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the S_x expectation value list."""
        return self.jx.shape
