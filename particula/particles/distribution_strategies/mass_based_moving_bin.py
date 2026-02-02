"""Mass-based moving bin distribution strategy.

This module defines :class:`MassBasedMovingBin`, a
:class:`~.base.DistributionStrategy` implementation that keeps total mass per
bin and replicates species-aware interfaces when the underlying data is not
per-species.

Example:
    >>> from particula.particles.distribution_strategies import (
    ...     MassBasedMovingBin,
    ... )
    >>> strategy = MassBasedMovingBin()
    >>> strategy.get_name()
    "MassBasedMovingBin"
"""

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import DistributionStrategy

logger = logging.getLogger("particula")


class MassBasedMovingBin(DistributionStrategy):
    """Mass-based moving bin distribution strategy.

    This strategy stores the total mass per bin but exposes per-species
    interfaces to downstream consumers that expect (n_particles, n_species)
    inputs. Mass additions and concentration updates always operate on the
    bin-level mass, while helpers replicate or average values when species are
    involved.

    Attributes:
        logger: Module-scoped logger that records strategy warnings.

    Example:
        >>> strategy = MassBasedMovingBin()
        >>> strategy.get_name()
        "MassBasedMovingBin"
    """

    def get_species_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate the mass assigned to each species.

        The mass-based strategy stores the total mass per bin. When the density
        array includes multiple species, the total mass is duplicated across
        species so hungry consumers receive (n_particles, n_species) arrays.

        Args:
            distribution: Total mass per bin array with either one or two axes.
            density: Species density array determining the replicating axis
                when the species count exceeds one.

        Returns:
            An array shaped (n_particles, n_species) representing per-species
            mass values.
        """
        distribution_arr = np.asarray(distribution, dtype=np.float64)
        if distribution_arr.ndim == 1:
            species_axis = distribution_arr[:, np.newaxis]
            if density.size == 1:
                return species_axis
            return np.tile(species_axis, (1, density.size))
        return distribution_arr

    def get_mass(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the mass for each particle in the bin strategy.

        When the distribution is one-dimensional, each entry already represents
        the per-particle mass and can be returned directly. If a second axis
        exists, delegate to :meth:`DistributionStrategy.get_mass` to sum across
        species and preserve existing behavior.

        Args:
            distribution: Mass distribution array that may include species.
            density: Density array used by the base implementation when a
                species axis must be collapsed.

        Returns:
            A one-dimensional array of mass per particle values.
        """
        distribution_arr = np.asarray(distribution, dtype=np.float64)
        if distribution_arr.ndim == 1:
            return distribution_arr
        return super().get_mass(distribution_arr, density)

    def get_radius(
        self, distribution: NDArray[np.float64], density: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate particle radius from individual mass values.

        Args:
            distribution: Mass distribution array used to infer volume.
            density: Density array used to convert mass to volume.

        Returns:
            Particle radius array in meters computed from the mass-to-volume
            relationship.
        """
        volumes = distribution / density
        return (3 * volumes / (4 * np.pi)) ** (1 / 3)

    def add_mass(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        density: NDArray[np.float64],
        added_mass: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Add mass to the distribution bins and keep the concentration.

        This strategy increases the total mass per bin, leaving concentration
        untouched because the moving bin already represents mass-based
        concentrations.

        Args:
            distribution: Existing mass distribution per bin.
            concentration: Particle number concentration per bin.
            density: Species density array (unused but included for interface
                compatibility).
            added_mass: Mass increments to add to each bin.

        Returns:
            A tuple containing the updated distribution and concentration
            arrays.
        """
        return distribution + added_mass, concentration

    def add_concentration(  # pylint: disable=R0801
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
        """Add concentration to the bins while optionally averaging charge.

        Args:
            distribution: Current mass distribution per bin.
            concentration: Existing concentration per bin.
            added_distribution: Mass distribution for the added material.
            added_concentration: Concentration associated with the addition.
            charge: Optional charge to average over concentration-weighted bins.
            added_charge: Optional charge attached to the added material.

        Returns:
            A tuple of updated distribution, concentration, and charge arrays.

        Raises:
            ValueError: When distribution or concentration shapes mismatch the
                added arrays, or when charge shapes disagree with concentration.
        """
        if (distribution.shape != added_distribution.shape) or (
            not np.allclose(distribution, added_distribution, rtol=1e-6)
        ):
            message = (
                "When adding concentration to MassBasedMovingBin, "
                "distribution and added distribution should match."
            )
            logger.error(message)
            raise ValueError(message)
        if concentration.shape != added_concentration.shape:
            message = (
                "When adding concentration to MassBasedMovingBin, the arrays "
                "should have the same shape."
            )
            logger.error(message)
            raise ValueError(message)

        original_concentration = concentration.copy()
        concentration += added_concentration

        if charge is None:
            return distribution, concentration, None

        # Handle scalar charge (e.g., charge=0) by converting to array
        if not isinstance(charge, np.ndarray):
            charge = np.full_like(
                original_concentration, charge, dtype=np.float64
            )

        if charge.shape != original_concentration.shape:
            message = (
                "When adding concentration with charge, charge must match "
                "concentration shape."
            )
            logger.error(message)
            raise ValueError(message)

        if added_charge is None:
            return distribution, concentration, charge

        if added_charge.shape != added_concentration.shape:
            message = (
                "When adding concentration with charge, added_charge must "
                "match added_concentration shape."
            )
            logger.error(message)
            raise ValueError(message)

        total_concentration = original_concentration + added_concentration
        numerator = (
            charge * original_concentration + added_charge * added_concentration
        )
        # Weighted average with zero-bin fallback to added_charge to avoid NaN.
        updated_charge = np.divide(
            numerator,
            total_concentration,
            out=np.zeros_like(total_concentration, dtype=np.float64),
            where=total_concentration != 0,
        )
        updated_charge = np.where(
            total_concentration == 0, added_charge, updated_charge
        )
        return distribution, concentration, updated_charge

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
        """Indicate that pairwise collisions are unsupported.

        Arguments:
            distribution: Mass distribution array.
            concentration: Concentration array for the bins.
            density: Density array used for radius/mass conversions.
            indices: Pairwise collision indices of shape (K, 2).
            charge: Optional charge array involved in the collision.

        Raises:
            NotImplementedError: Always raised since MassBasedMovingBin does not
                represent discrete particles but aggregate mass bins.
        """
        message = (
            "Colliding pairs in MassBasedMovingBin is not physically meaningful"
        )
        logger.warning(message)
        raise NotImplementedError(message)
