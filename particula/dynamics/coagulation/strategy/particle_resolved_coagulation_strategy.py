"""
Particle-resolved coagulation strategy class. This class implements the
methods defined in the CoagulationStrategy abstract class.
"""

# from typing import Union, Optional
# import logging
# import numpy as np
# from numpy.typing import NDArray

# from particula.particles.representation import ParticleRepresentation
# from particula.dynamics.coagulation.strategy.coagulation_strategy_abc import (
#     CoagulationStrategyABC,
# )
# from particula.dynamics.coagulation.brownian_kernel import (
#     get_brownian_kernel_via_system_state,
# )

# logger = logging.getLogger("particula")


# class ParticleResolvedCoagulationStrategy(CoagulationStrategyABC):
#     """
#     Particle-resolved coagulation strategy class. This class implements the
#     methods defined in the CoagulationStrategy abstract class. The kernel
#     strategy is passed as an argument to the class, should use a dimensionless
#     kernel representation.

#     Parameters:
#         - distribution_type : The type of distribution to be used with the
#             coagulation strategy. Must be "particle_resolved".
#         - kernel_radius : The kernel radius for the particle [m].
#         - kernel_bins_number : The number of kernel bins for the particle
#             [dimensionless].
#         - kernel_bins_per_decade : The number of kernel bins per decade
#             [dimensionless].

#     Methods:
#         kernel: Calculate the coagulation kernel.
#         step: Perform a single step of the coagulation process.
#     """

#     def __init__(
#         self,
#         distribution_type: str,
#         kernel_radius: Optional[NDArray[np.float64]] = None,
#         kernel_bins_number: Optional[int] = None,
#         kernel_bins_per_decade: int = 10,
#     ):
#         if distribution_type != "particle_resolved":
#             raise ValueError(
#                 "Invalid distribution type. "
#                 "Must be 'particle_resolved' for "
#                 "`ParticleResolvedCoagulationStrategy`."
#             )
#         CoagulationStrategyABC.__init__(
#             self, distribution_type=distribution_type
#         )
#         self.kernel_radius = kernel_radius
#         self.kernel_bins_number = kernel_bins_number
#         self.kernel_bins_per_decade = kernel_bins_per_decade

#     def get_kernel_radius(
#         self, particle: ParticleRepresentation
#     ) -> NDArray[np.float64]:
#         """Get the binning for the kernel radius.

#         If the kernel radius is not set, it will be calculated based on the
#         particle radius.

#         Args:
#             particle : The particle for which the kernel radius is to be
#                 calculated.

#         Returns:
#             The kernel radius for the particle [m].
#         """
#         if self.kernel_radius is not None:
#             return self.kernel_radius
#         # else find the non-zero min and max radii, the log space them
#         min_radius = np.min(particle.get_radius()[particle.get_radius() > 0])
#         max_radius = np.max(particle.get_radius()[particle.get_radius() > 0])
#         if self.kernel_bins_number is not None:
#             return np.logspace(
#                 np.log10(min_radius),
#                 np.log10(max_radius),
#                 num=self.kernel_bins_number,
#                 base=10,
#                 dtype=np.float64,
#             )
#         # else kernel bins per decade
#         num = np.ceil(
#             self.kernel_bins_per_decade * np.log10(max_radius / min_radius),
#         )
#         return np.logspace(
#             np.log10(min_radius),
#             np.log10(max_radius),
#             num=int(num),
#             base=10,
#             dtype=np.float64,
#         )

#     def dimensionless_kernel(
#         self,
#         diffusive_knudsen: NDArray[np.float64],
#         coulomb_potential_ratio: NDArray[np.float64],
#     ) -> NDArray[np.float64]:

#         message = (
#             "Dimensionless kernel not implemented in particle-resolved "
#             + "coagulation strategy."
#         )
#         logger.error(message)
#         raise NotImplementedError(message)

#     def kernel(
#         self,
#         particle: ParticleRepresentation,
#         temperature: float,
#         pressure: float,
#     ) -> Union[float, NDArray[np.float64]]:
#         # need to update later with the correct mass dependency
#         radius_bins = self.get_kernel_radius(particle)
#         mass_bins = 4 / 3 * np.pi * np.power(radius_bins, 3) * 1000  # type: ignore
#         return get_brownian_kernel_via_system_state(
#             particle_radius=radius_bins,  # type: ignore
#             mass_particle=mass_bins,  # type: ignore
#             temperature=temperature,
#             pressure=pressure,
#         )

#     def step(
#         self,
#         particle: ParticleRepresentation,
#         temperature: float,
#         pressure: float,
#         time_step: float,
#     ) -> ParticleRepresentation:

#         loss_gain_indices = get_particle_resolved_coagulation_step(
#             particle_radius=particle.get_radius(),
#             kernel=self.kernel(  # type: ignore
#                 particle=particle, temperature=temperature, pressure=pressure
#             ),
#             kernel_radius=self.get_kernel_radius(particle),
#             volume=particle.volume,
#             time_step=time_step,
#             random_generator=np.random.default_rng(),
#         )
#         particle.collide_pairs(loss_gain_indices)
#         return particle
