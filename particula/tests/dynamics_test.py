""" test the simple solver
"""

import pytest
import numpy as np
from particula import particle, dynamics

some_kwargs = {
    "mode": 200e-9,  # 200 nm median
    "nbins": 1000,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.2,  # relatively narrow
    "cutoff": .99999,  # let's take it all lol
}

particle_dist = particle.Particle(**some_kwargs)

Solver = dynamics.Solver(
    particle=particle_dist,
    time_span=np.linspace(0, 100, 200),
)

FinerSolver = dynamics.Solver(
    particle=particle_dist,
    time_span=np.linspace(0, 100, 500),
)

radius = particle_dist.particle_radius.m
solution = Solver.solution()
fine_sols = FinerSolver.solution()


def test_solution():
    """ test fidelity of the solution
    """

    assert solution.m.shape == (200, 1000)
    assert solution.u == particle_dist.particle_distribution().u
    assert fine_sols.m.shape == (500, 1000)
    assert (solution.m[0, :] == fine_sols.m[0, :]).all()


def test_conservation():
    """ test conservation of mass/volume
    """

    assert (
        np.trapz((solution.m[0, :]-solution.m[-1, :])*radius**3, radius) /
        np.trapz((solution.m[0, :])*radius**3, radius)
        ==
        pytest.approx(0, abs=1e-4)
    )

    assert (
        np.trapz((fine_sols.m[0, :]-fine_sols.m[-1, :])*radius**3, radius) /
        np.trapz((fine_sols.m[0, :])*radius**3, radius)
        ==
        pytest.approx(0, abs=1e-4)
    )
