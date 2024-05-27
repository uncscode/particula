"""Test Particles and Strategies."""

# import numpy as np
# import pytest
# from particula.next.particles.distribution_strategies import (
#     MassBasedMovingBin, RadiiBasedMovingBin, SpeciatedMassMovingBin,
# )
# from particula.next.particles.surface_strategies import SurfaceStrategyVolume
# from particula.next.particles.activity_strategies import IdealActivityMass


# mass_based_strategy = MassBasedMovingBin()
# radii_based_strategy = RadiiBasedMovingBin()
# speciated_mass_strategy = SpeciatedMassMovingBin()
# surface_strategy = SurfaceStrategyVolume()
# activity_strategy = IdealActivityMass()



# @pytest.mark.parametrize("strategy, distribution, density, concentration", [
#     (MassBasedMovingBin(),
#      np.array([100, 200, 300], dtype=np.float64),
#      np.float64(2.5),
#      np.array([10, 20, 30], dtype=np.float64)),
#     (RadiiBasedMovingBin(),
#      np.array([1, 2, 3], dtype=np.float64),
#      np.float64(5), np.array([10, 20, 30], dtype=np.float64)),
#     # For SpeciatedMassStrategy, ensure distribution aligns with expected 2D
#     # shape and densities are properly set
#     (SpeciatedMassMovingBin(),
#      np.array([[100, 200], [300, 400]], dtype=np.float64),
#      np.array([2.5, 3.5], dtype=np.float64),
#      np.array([10, 20], dtype=np.float64)),
# ])
# def test_particle_properties(strategy, distribution, density, concentration):
#     """Parameterized test for Particle properties."""
#     particle = Particle(strategy, activity_strategy, surface,
#                         distribution, density, concentration)
#     mass = particle.get_mass()
#     radius = particle.get_radius()
#     total_mass = particle.get_total_mass()

#     # Validate the types of the returned values
#     assert isinstance(mass, np.ndarray)
#     assert isinstance(radius, np.ndarray)
#     assert isinstance(total_mass, np.float_)
#     # The value of the returned mass, radius, and total_mass should be correct
#     # they are already tested in the strategy tests
