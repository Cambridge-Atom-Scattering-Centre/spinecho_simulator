"""Module for representing and manipulating spin states.

In this package, spin states are represented by the `Spin` class,
which encapsulates the properties of a spin.
An example of its useage can be seen below.

.. literalinclude:: ../../examples/spin_representation.py
    :language: python
    :lineno-start: 8
    :lines: 8-43
    :dedent: 4

"""

from __future__ import annotations

from spinecho_sim.state._displacement import (
    ParticleDisplacement,
    ParticleDisplacementList,
)
from spinecho_sim.state._samples import (
    sample_constant_displacement,
    sample_constant_velocity,
    sample_gaussian_velocities,
    sample_uniform_displacement,
)
from spinecho_sim.state._spin import (
    CoherentSpin,
    CoherentSpinList,
    EmptySpin,
    EmptySpinList,
    EmptySpinListList,
    GenericSpin,
    GenericSpinList,
    Spin,
    get_bargmann_expectation_values,
    get_expectation_values,
)
from spinecho_sim.state._state import (
    DiatomicParticleState,
    MonatomicParticleState,
)
from spinecho_sim.state._trajectory import (
    DiatomicTrajectory,
    DiatomicTrajectoryList,
    MonatomicTrajectory,
    MonatomicTrajectoryList,
)

__all__ = [
    "CoherentSpin",
    "CoherentSpinList",
    "DiatomicParticleState",
    "DiatomicTrajectory",
    "DiatomicTrajectoryList",
    "EmptySpin",
    "EmptySpinList",
    "EmptySpinListList",
    "GenericSpin",
    "GenericSpinList",
    "MonatomicParticleState",
    "MonatomicTrajectory",
    "MonatomicTrajectoryList",
    "ParticleDisplacement",
    "ParticleDisplacementList",
    "Spin",
    "get_bargmann_expectation_values",
    "get_expectation_values",
    "sample_constant_displacement",
    "sample_constant_velocity",
    "sample_gaussian_velocities",
    "sample_uniform_displacement",
]
