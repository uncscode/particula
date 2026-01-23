"""Particle-resolved speciated mass distribution strategy.

This module defines a strategy where each particle maintains mass per
species along with optional charge. It handles concentration updates,
coagulation, and charge conservation while remaining compatible with the
particle-resolved kernel framework.
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class ParticleResolvedSpeciatedMass(DistributionStrategy):
    """Represent particle-resolved mass for multiple species.

    Each particle maintains per-species masses and optional charge, enabling
    detailed tracking of condensation or coagulation events in particle-resolved
    workflows.

    Attributes:
        None explicitly stored; the strategy operates purely on arrays passed to
        its methods.

    Methods:
        get_species_mass: Return mass distribution per species.
        get_radius: Compute particle radius from mass and density.
        add_mass: Update particle distribution when mass is added.
        add_concentration: Extend the distribution with new particles.
        collide_pairs: Merge mass/concentration when particles coagulate.
    """

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the per-species mass array for each particle.

        Args:
            distribution: Mass per particle array with shape ``(N,)`` or
                ``(N, M)`` depending on the species count.
            density: Species densities used when converting between mass and
                volume (ignored for this strategy but part of the interface).

        Returns:
            NDArray[np.float64]: Array of per-species masses identical to
                ``distribution``.
        """
        return distribution

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute particle radius from mass and density.

        Args:
            distribution: Mass array for each particle and species.
            density: Per-species densities used to convert mass to volume.

        Returns:
            NDArray[np.float64]: Radius in metres for each particle.
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

        Args:
            distribution: Current mass distribution per particle and species.
            concentration: Concentration of each particle or bin.
            density: Species densities used for consistency with the interface.
            added_mass: Mass change to apply per particle.

        Returns:
            Tuple of the updated distribution and concentration arrays.
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

    def add_concentration(  # pylint: disable=too-many-branches  # noqa: C901
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        added_distribution: NDArray[np.float64],
        added_concentration: NDArray[np.float64],
        charge: Optional[NDArray[np.float64]] = None,
        added_charge: Optional[NDArray[np.float64]] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        Optional[NDArray[np.float64]],
    ]:
        """Add new particles and optional charge to the distribution.

        Args:
            distribution: Existing mass distribution array.
            concentration: Existing concentration per particle or bin.
            added_distribution: Mass distribution of arriving particles.
            added_concentration: Concentration of arriving particles.
            charge: Optional charge array for current particles.
            added_charge: Optional charge array for arriving particles.

        Returns:
            Tuple of distribution, concentration, and charge arrays after update.
        """

        added_distribution = np.atleast_1d(added_distribution)
        added_concentration = np.atleast_1d(added_concentration)
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

        # Handle charge defaults and validation.

        charge_added = added_charge

        if charge is not None:
            charge = np.atleast_1d(charge)
            if charge_added is None:
                # Default new particle charges to zero when not provided.
                charge_added = np.zeros_like(added_concentration)
            else:
                charge_added = np.atleast_1d(charge_added)
            if charge_added.shape != added_concentration.shape:
                message = (
                    "When adding concentration with charge, added_charge "
                    "must match added_concentration shape."
                )
                logger.error(message)
                raise ValueError(message)

        empty_bins = np.flatnonzero(concentration == 0)
        empty_bins_count = len(empty_bins)
        added_bins_count = len(added_concentration)
        if empty_bins_count >= added_bins_count:
            distribution[empty_bins[:added_bins_count]] = added_distribution
            concentration[empty_bins[:added_bins_count]] = added_concentration
            if charge is not None and charge_added is not None:
                charge[empty_bins[:added_bins_count]] = charge_added
            return distribution, concentration, charge
        if empty_bins_count > 0:
            distribution[empty_bins] = added_distribution[:empty_bins_count]
            concentration[empty_bins] = added_concentration[:empty_bins_count]
            if charge is not None and charge_added is not None:
                charge[empty_bins] = charge_added[:empty_bins_count]
        distribution = np.concatenate(
            (distribution, added_distribution[empty_bins_count:]), axis=0
        )
        concentration = np.concatenate(
            (concentration, added_concentration[empty_bins_count:]), axis=0
        )
        if charge is None:
            return distribution, concentration, None
        if charge_added is not None:
            charge = np.concatenate(
                (charge, charge_added[empty_bins_count:]),
                axis=0,
            )
        return distribution, concentration, charge

    def collide_pairs(  # pylint: disable=too-many-positional-arguments
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        indices: NDArray[np.int64],
        charge: Optional[NDArray[np.float64]] = None,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]
    ]:
        """Merge mass, concentration, and charge for collided particle pairs.

        Args:
            distribution: Mass distribution array. Shape ``(N,)`` for single
                species or ``(N, M)`` for multiple species.
            concentration: Concentration array of shape ``(N,)``.
            density: Species density array of shape ``(M,)``.
            indices: Collision pairs where each row is ``[small_index,
                large_index]`` describing the merge direction.
            charge: Optional charge array. When provided, charges are conserved
                by summing the colliding pair charges.

        Returns:
            Tuple of updated distribution, concentration, and optional charge
            arrays.
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

        # Handle charge if present as numpy array and non-zero
        # charge can be None or array - only process if array
        if charge is not None and isinstance(charge, np.ndarray):
            # Check only colliding pairs for non-zero charges (performance opt)
            if np.any(charge[small_index] != 0) or np.any(
                charge[large_index] != 0
            ):
                charge[large_index] += charge[small_index]
                charge[small_index] = 0

        return distribution, concentration, charge
