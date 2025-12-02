"""Particle resolved speciated mass strategy."""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class ParticleResolvedSpeciatedMass(DistributionStrategy):
    """Strategy for particle-resolved masses with multiple species.

    Allows each particle to have separate masses for each species, with
    individualized densities. This strategy provides a more detailed
    approach when each particle's composition must be modeled explicitly.

    Methods:
    - get_name : Return the type of the distribution strategy.
    - get_species_mass : Calculate the mass per species.
    - get_mass : Calculate the mass of the particles or bin.
    - get_total_mass : Calculate the total mass of particles.
    - get_radius : Calculate the radius of particles.
    - add_mass : Add mass to the particle distribution.
    - add_concentration : Add concentration to the distribution.
    - collide_pairs : Perform collision logic on specified particle pairs.
    """

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate the mass per species for each particle.

        Returns:
            Mass per species array for each particle.
        """
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate particle radius from multi-species mass and density.

        Returns:
            Particle radius in meters for each particle.
        """
        if distribution.ndim == 1:
            volumes = distribution / density
        else:
            volumes = np.sum(distribution / density, axis=1)
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(  # pylint: disable=R0801
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add mass to individual particles in the distribution.

        Returns:
            Updated distribution and concentration arrays.
        """
        if distribution.ndim == 2:
            concentration_expand = concentration[:, np.newaxis]
        else:
            concentration_expand = concentration
        new_mass = np.divide(
            np.maximum(distribution * concentration_expand + added_mass, 0),
            concentration_expand,
            out=np.zeros_like(distribution),
            where=concentration_expand != 0,
        )
        if new_mass.ndim == 1:
            new_mass_sum = np.sum(new_mass)
        else:
            new_mass_sum = np.sum(new_mass, axis=1)
        concentration = np.where(new_mass_sum > 0, concentration, 0)
        return new_mass, concentration

    def add_concentration(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add new particles to the distribution.

        Returns:
            Updated distribution and concentration arrays.
        """
        rescaled = False
        if np.all(added_concentration == 1):
            rescaled = True
        max_concentration = np.max(concentration)
        if np.allclose(
            added_concentration, max_concentration, atol=1e-2
        ) or np.all(concentration == 0):
            if max_concentration > 0:
                added_concentration = added_concentration / max_concentration
            rescaled = True
        if not rescaled:
            message = (
                "When adding concentration to ParticleResolvedSpeciatedMass, "
                "added concentration should be all ones or all the same."
            )
            logger.error(message)
            raise ValueError(message)

        concentration = np.divide(
            concentration,
            concentration,
            out=np.zeros_like(concentration),
            where=concentration != 0,
        )

        empty_bins = np.flatnonzero(concentration == 0)
        empty_bins_count = len(empty_bins)
        added_bins_count = len(added_concentration)
        if empty_bins_count >= added_bins_count:
            distribution[empty_bins] = added_distribution
            concentration[empty_bins] = added_concentration
            return distribution, concentration
        if empty_bins_count > 0:
            distribution[empty_bins] = added_distribution[:empty_bins_count]
            concentration[empty_bins] = added_concentration[:empty_bins_count]
        distribution = np.concatenate(
            (distribution, added_distribution[empty_bins_count:]), axis=0
        )
        concentration = np.concatenate(
            (concentration, added_concentration[empty_bins_count:]), axis=0
        )
        return distribution, concentration

    def collide_pairs(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
        charge: Optional[NDArray[np.float64]] = None,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]
    ]:
        """Collide specified particle pairs by merging mass and charge.

        Performs coagulation between particle pairs for particle-resolved
        simulations. The smaller particle's mass is added to the larger
        particle, and the smaller particle's concentration is set to zero.
        If a charge array is provided, charges are conserved by summing the
        charges of the colliding pair.

        The charge handling is optimized: charges are only processed when the
        charge array is provided as a numpy array AND at least one of the
        colliding particles has a non-zero charge.

        Arguments:
            - distribution : The mass distribution array. Shape is (N,) for
                single species or (N, M) for M species per particle.
            - concentration : The concentration array of shape (N,).
            - density : The density array of shape (M,) for species densities.
            - indices : Collision pair indices array of shape (K, 2) where
                each row is [small_index, large_index].
            - charge : Optional charge array of shape (N,). If provided and
                contains non-zero values in colliding pairs, charges will be
                summed during collisions. If None, charge handling is skipped.

        Returns:
            A tuple containing:
                - Updated distribution array with merged masses.
                - Updated concentration array with zeroed small particles.
                - Updated charge array (None if input was None).
        """
        small_index = indices[:, 0]
        large_index = indices[:, 1]

        # Handle mass (existing logic)
        if distribution.ndim == 1:
            distribution[large_index] += distribution[small_index]
            distribution[small_index] = 0
        else:
            distribution[large_index, :] += distribution[small_index, :]
            distribution[small_index, :] = 0
        concentration[small_index] = 0

        # Handle charge if present as an array and non-zero in colliding pairs
        # charge can be None, 0 (int), or an array - only process if array
        if charge is not None and isinstance(charge, np.ndarray):
            # Check only colliding pairs for non-zero charges (performance opt)
            colliding_charges = np.concatenate(
                [charge[small_index], charge[large_index]]
            )
            if np.any(colliding_charges != 0):
                charge[large_index] += charge[small_index]
                charge[small_index] = 0

        return distribution, concentration, charge
