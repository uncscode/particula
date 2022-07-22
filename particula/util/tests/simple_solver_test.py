""" test the simple solver
"""

import pytest
import numpy as np
from particula import particle, dynamic_step
from particula.util import simple_solver

simple_dic_kwargs = {
    "mode": 200e-9,  # 200 nm median
    "nbins": 1000,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.2,  # relatively narrow
    "cutoff": .99999,  # let's take it all lol
}

particle_dist = particle.Particle(**simple_dic_kwargs)

coag_kern = dynamic_step.DynamicStep(**simple_dic_kwargs).coag_kern()

Solver = simple_solver.SimpleSolver(
    distribution=particle_dist.particle_distribution(),
    radius=particle_dist.particle_radius,
    kernel=coag_kern,
    tspan=np.linspace(0, 100, 100),
)

FinerSolver = simple_solver.SimpleSolver(
    distribution=particle_dist.particle_distribution(),
    radius=particle_dist.particle_radius,
    kernel=coag_kern,
    tspan=np.linspace(0, 100, 1000),
)

radius = particle_dist.particle_radius.m
solution = Solver.solution()
fine_sols = FinerSolver.solution()


def test_dims():
    """ test the dimensions
    """

    assert (
        Solver.prep_inputs()[0].shape ==
        particle_dist.particle_distribution().m.shape
    )
    assert (
        Solver.prep_inputs()[1].shape ==
        particle_dist.particle_radius.m.shape
    )
    assert (
        Solver.prep_inputs()[2].shape ==
        coag_kern.m.shape
    )
    assert (
        Solver.prep_inputs()[3].shape ==
        np.linspace(0, 100, 100).shape
    )


def test_solution():
    """ test the solution
    """

    assert solution.m.shape == (100, 1000)
    assert solution.u == particle_dist.particle_distribution().u
    assert fine_sols.m.shape == (1000, 1000)
    assert (solution.m[0, :] == fine_sols.m[0, :]).all()


def test_conservation():
    """ test the conservation
    """

    assert (
        np.trapz((solution.m[0, :]-solution.m[-1, :])*radius**3, radius) /
        np.trapz((solution.m[0, :])*radius**3, radius)
        ==
        pytest.approx(0, abs=1e-3)
    )

    assert (
        np.trapz((fine_sols.m[0, :]-fine_sols.m[-1, :])*radius**3, radius) /
        np.trapz((fine_sols.m[0, :])*radius**3, radius)
        ==
        pytest.approx(0, abs=1e-3)
    )
