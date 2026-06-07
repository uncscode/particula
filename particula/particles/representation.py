"""Legacy particle representation facade.

Provides a deprecated facade over :class:`ParticleData` while preserving the
legacy API for distribution strategies, activities, and surfaces.
"""

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

# From Particula
from particula.particles.activity_strategies import ActivityStrategy
from particula.particles.distribution_strategies import (
    DistributionStrategy,
)
from particula.particles.properties.sort_bins import (
    get_sorted_bins_by_radius,
)
from particula.particles.surface_strategies import SurfaceStrategy

logger = logging.getLogger("particula")

_DEPRECATION_MESSAGE = (
    "ParticleRepresentation is deprecated. Use ParticleData instead. "
    "See migration guide: docs/Features/particle-data-migration.md"
)


def _normalize_density_array(
    density: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Normalize density input to a 1D float64 array.

    Args:
        density: Density specified as a scalar or 1D array.

    Returns:
        Density as a 1D ``float64`` array.

    Raises:
        ValueError: If ``density`` cannot be reduced to a scalar or 1D array.
    """
    density_array = np.asarray(density, dtype=np.float64)
    if density_array.ndim == 0:
        return np.atleast_1d(density_array)
    if density_array.ndim == 1:
        return density_array
    if density_array.size == 0:
        return np.zeros(density_array.shape[-1], dtype=np.float64)
    raise ValueError(
        "density must be a scalar or 1D array, "
        f"got array with shape {density_array.shape}"
    )


def _normalize_charge_array(
    charge: NDArray[np.float64] | float,
    template: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize charge input to exactly match the template shape.

    Args:
        charge: Charge specified as a scalar or array-like value.
        template: Reference array defining the required output shape.

    Returns:
        Charge as a ``float64`` array matching ``template``.

    Raises:
        ValueError: If ``charge`` cannot be broadcast to ``template``.
    """
    charge_array = np.asarray(charge, dtype=np.float64)
    if charge_array.ndim == 0:
        return np.full_like(template, float(charge_array), dtype=np.float64)
    if charge_array.size == 0:
        return np.zeros_like(template, dtype=np.float64)
    if charge_array.shape == template.shape:
        return charge_array
    raise ValueError(
        "charge must be a scalar, empty array, or array matching concentration "
        f"shape {template.shape}; got shape {charge_array.shape}"
    )


if TYPE_CHECKING:
    from particula.particles.particle_data import ParticleData


def _warn_deprecated(*, stacklevel: int = 2) -> None:
    """Log a deprecation notice at INFO level.

    Uses ``logger.info`` instead of ``warnings.warn`` so the message is
    visible during normal operation but never triggers failures under
    ``-Werror`` or ``pytest -W error``.

    Args:
        stacklevel: Unused, kept for call-site compatibility.
    """
    logger.info(_DEPRECATION_MESSAGE)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class ParticleRepresentation:
    """Everything needed to represent a particle or a collection of particles.

    Attributes:
        strategy: Distribution strategy for particle representation.
        activity: Activity strategy for partial pressure calculations.
        surface: Surface strategy for surface tension/Kelvin effects.
        distribution: Distribution data for the particles.
        density: Density array for the particles.
        concentration: Concentration data for the particles.
        charge: Charge per particle.
        volume: Simulation volume for the representation.
    """

    _data: "ParticleData"
    _distribution: NDArray[np.float64]
    _charge_value: NDArray[np.float64] | float | None
    _charge_array: NDArray[np.float64] | None
    _charge_is_none: bool
    _species_mass_is_1d: bool

    def __init__(
        self,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64] | float,
        density: NDArray[np.float64] | float,
        concentration: NDArray[np.float64] | float,
        charge: NDArray[np.float64] | float | None,
        volume: float = 1,
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        """Initialize the legacy particle representation facade.

        Args:
            strategy: Distribution strategy for the representation.
            activity: Activity strategy for particle-phase calculations.
            surface: Surface strategy for surface-property calculations.
            distribution: Distribution data interpreted by ``strategy``.
            density: Particle density values.
            concentration: Particle concentration values.
            charge: Optional particle charge values.
            volume: Simulation volume in m^3. Defaults to 1.

        Notes:
            This legacy facade emits a deprecation log message and stores its
            state in an underlying :class:`ParticleData` container.
        """
        _warn_deprecated(stacklevel=2)

        self.strategy = strategy
        self.activity = activity
        self.surface = surface

        self._distribution = np.asarray(distribution, dtype=np.float64)
        density_array = _normalize_density_array(density)
        concentration_array = np.asarray(concentration, dtype=np.float64)
        if concentration_array.ndim == 0:
            concentration_array = np.atleast_1d(concentration_array)
        if self._distribution.ndim > 1 and concentration_array.ndim == 1:
            if concentration_array.size == 1:
                concentration_array = np.full(
                    self._distribution.shape[0],
                    concentration_array.item(),
                    dtype=np.float64,
                )
        self._charge_is_none = charge is None
        self._charge_value = None
        if charge is not None:
            self._charge_value = self._coerce_array(
                charge,
                concentration_array,
            )
        self._charge_array = None
        self._species_mass_is_1d = False

        (
            self._data,
            self._species_mass_is_1d,
            self._charge_array,
        ) = self._build_data(
            distribution=self._distribution,
            density_array=density_array,
            concentration_array=concentration_array,
            charge_value=None if self._charge_is_none else self._charge_value,
            volume_value=float(volume),
            strategy=self.strategy,
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the representation.

        Returns:
            Multi-line string describing the configured strategies and bulk
            concentrations.
        """
        return (
            f"Particle Representation:\n"
            f"\tStrategy: {self.get_strategy_name()}\n"
            f"\tActivity: {self.get_activity_name()}\n"
            f"\tSurface: {self.get_surface_name()}\n"
            f"\tMass Concentration: "
            f"{self.get_mass_concentration():.3e} [kg/m^3]\n"
            f"\tNumber Concentration: "
            f"{self.get_total_concentration():.3e} [#/m^3]"
        )

    @staticmethod
    def _coerce_array(
        value: NDArray[np.float64] | float,
        template: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a float64 array broadcast to the template shape.

        Args:
            value: Scalar or array-like value to normalize.
            template: Reference array defining the required output shape.

        Returns:
            Normalized ``float64`` array matching ``template``.
        """
        return _normalize_charge_array(value, template)

    @property
    def data(self) -> "ParticleData":
        """Return the underlying ParticleData container."""
        return self._data

    @classmethod
    def from_data(
        cls,
        data: "ParticleData",
        *,
        strategy: DistributionStrategy,
        activity: ActivityStrategy,
        surface: SurfaceStrategy,
        distribution: NDArray[np.float64],
        charge: NDArray[np.float64] | None = None,
    ) -> "ParticleRepresentation":
        """Create a facade without emitting a deprecation warning.

        Args:
            data: ParticleData instance to wrap.
            strategy: Distribution strategy for this facade.
            activity: Activity strategy for this facade.
            surface: Surface strategy for this facade.
            distribution: Cached distribution values for the facade.
            charge: Optional charge array to preserve legacy API.

        Returns:
            ParticleRepresentation instance wrapping ``data``.
        """
        instance = cls.__new__(cls)
        instance.strategy = strategy
        instance.activity = activity
        instance.surface = surface
        instance._data = data
        instance._distribution = np.asarray(distribution, dtype=np.float64)
        instance._charge_value = None
        instance._charge_array = None
        if charge is not None:
            instance._charge_value = np.asarray(charge, dtype=np.float64)
            instance._charge_array = instance._charge_value
        instance._species_mass_is_1d = data.masses.shape[-1] == 1
        instance._charge_is_none = instance._charge_array is None
        return instance

    @property
    def distribution(self) -> NDArray[np.float64]:
        """Return the cached distribution array."""
        return self._distribution

    @distribution.setter
    def distribution(self, value: NDArray[np.float64]) -> None:
        _warn_deprecated(stacklevel=2)
        new_distribution = np.asarray(value, dtype=np.float64)
        self._update_state(
            distribution=new_distribution,
            concentration=self._data.concentration[0],
            charge_array=self._charge_array,
        )

    @property
    def density(self) -> NDArray[np.float64]:
        """Return the density array."""
        return self._data.density

    @property
    def concentration(self) -> NDArray[np.float64]:
        """Return the concentration array for box 0."""
        return self._data.concentration[0]

    @concentration.setter
    def concentration(self, value: NDArray[np.float64]) -> None:
        _warn_deprecated(stacklevel=2)
        self._data.concentration[0] = self._coerce_array(
            value,
            self._data.concentration[0],
        )

    @property
    def charge(self) -> NDArray[np.float64] | None:
        """Return charge array or None when charge is disabled."""
        if self._charge_is_none:
            return None
        return self._charge_array

    @charge.setter
    def charge(self, value: NDArray[np.float64] | float | None) -> None:
        _warn_deprecated(stacklevel=2)
        if value is None:
            self._charge_is_none = True
            self._charge_value = None
            self._charge_array = None
            self._data.charge[0] = np.zeros_like(self._data.concentration[0])
            return
        self._charge_is_none = False
        charge_array = self._coerce_array(value, self._data.concentration[0])
        self._charge_value = charge_array
        self._charge_array = charge_array
        self._data.charge[0] = charge_array

    @property
    def volume(self) -> float:
        """Return the representation volume in m^3."""
        return float(self._data.volume[0])

    @volume.setter
    def volume(self, value: float) -> None:
        _warn_deprecated(stacklevel=2)
        self._data.volume[0] = float(value)

    @staticmethod
    def _build_masses(
        distribution: NDArray[np.float64],
        density_array: NDArray[np.float64],
        strategy: DistributionStrategy,
    ) -> tuple[NDArray[np.float64], bool]:
        """Build per-species masses from distribution data.

        Args:
            distribution: Distribution data for the representation.
            density_array: Density array for species.
            strategy: Strategy used to compute per-species masses.

        Returns:
            Tuple of mass array and flag indicating if the masses were 1D.
        """
        masses = strategy.get_species_mass(distribution, density_array)
        species_mass_is_1d = masses.ndim == 1
        if masses.ndim == 1:
            if density_array.size == masses.size:
                masses = np.tile(masses[:, np.newaxis], (1, masses.size))
            else:
                masses = masses[:, np.newaxis]
        return np.asarray(masses, dtype=np.float64), species_mass_is_1d

    @classmethod
    def _build_data(
        cls,
        *,
        distribution: NDArray[np.float64],
        density_array: NDArray[np.float64],
        concentration_array: NDArray[np.float64],
        charge_value: NDArray[np.float64] | float | None,
        volume_value: float,
        strategy: DistributionStrategy,
    ) -> tuple["ParticleData", bool, NDArray[np.float64] | None]:  # noqa: ANN201
        """Create particle data for a single-box representation.

        Args:
            distribution: Distribution data interpreted by ``strategy``.
            density_array: Normalized species-density array.
            concentration_array: Box concentration values.
            charge_value: Optional charge values to normalize.
            volume_value: Simulation volume in m^3.
            strategy: Distribution strategy used to derive mass data.

        Returns:
            Tuple containing the constructed particle data, whether the source
            species masses were 1D, and the normalized charge array.
        """
        from particula.particles.particle_data import ParticleData

        masses, species_mass_is_1d = cls._build_masses(
            distribution,
            density_array,
            strategy,
        )
        concentration_box = np.asarray(concentration_array, dtype=np.float64)[
            np.newaxis, ...
        ]
        charge_array = None
        if charge_value is not None:
            charge_array = _normalize_charge_array(
                charge_value,
                concentration_array,
            )
        charge_box = np.zeros_like(concentration_box)
        if charge_array is not None:
            charge_box = charge_array[np.newaxis, ...]
        volume_array = np.asarray([volume_value], dtype=np.float64)
        return (
            ParticleData(
                masses=np.asarray(masses[np.newaxis, ...], dtype=np.float64),
                concentration=concentration_box,
                charge=charge_box,
                density=density_array,
                volume=volume_array,
            ),
            species_mass_is_1d,
            charge_array,
        )

    def _update_state(
        self,
        distribution: NDArray[np.float64],
        concentration: NDArray[np.float64],
        charge_array: NDArray[np.float64] | None,
    ) -> None:
        """Update cached state from supplied arrays without warnings.

        Args:
            distribution: Updated distribution data.
            concentration: Updated concentration values.
            charge_array: Updated normalized charge data, if charge is tracked.
        """
        (
            self._data,
            self._species_mass_is_1d,
            self._charge_array,
        ) = self._build_data(
            distribution=distribution,
            density_array=self._data.density,
            concentration_array=concentration,
            charge_value=None if self._charge_is_none else charge_array,
            volume_value=float(self._data.volume[0]),
            strategy=self.strategy,
        )
        self._distribution = distribution
        if self._charge_is_none:
            self._charge_value = None
        else:
            self._charge_value = self._charge_array

    def get_strategy(self, clone: bool = False) -> DistributionStrategy:
        """Return the strategy used for particle representation.

        Args:
            clone: If ``True``, return a deep copy of the strategy.

        Returns:
            The distribution strategy used by this representation.
        """
        if clone:
            return deepcopy(self.strategy)
        return self.strategy

    def get_strategy_name(self) -> str:
        """Return the name of the strategy used for particle representation.

        Returns:
            The distribution strategy name.
        """
        return self.strategy.get_name()

    def get_activity(self, clone: bool = False) -> ActivityStrategy:
        """Return the activity strategy used for partial pressure calculations.

        Args:
            clone: If ``True``, return a deep copy of the activity strategy.

        Returns:
            The activity strategy used for particle-phase calculations.
        """
        if clone:
            return deepcopy(self.activity)
        return self.activity

    def get_activity_name(self) -> str:
        """Return the name of the activity strategy used for partial pressure
        calculations.

        Returns:
            The activity strategy name.
        """
        return self.activity.get_name()

    def get_surface(self, clone: bool = False) -> SurfaceStrategy:
        """Return surface strategy for surface tension and Kelvin effect.

        Args:
            clone: If ``True``, return a deep copy of the surface strategy.

        Returns:
            The surface strategy used for Kelvin and surface-tension effects.
        """
        if clone:
            return deepcopy(self.surface)
        return self.surface

    def get_surface_name(self) -> str:
        """Return the name of the surface strategy used for surface tension and
        Kelvin effect.

        Returns:
            The surface strategy name.
        """
        return self.surface.get_name()

    def get_distribution(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the distribution of the particles.

        Args:
            clone: If ``True``, return a copy of the distribution array.

        Returns:
            The particle distribution array.
        """
        if clone:
            return np.copy(self.distribution)
        return self.distribution

    def get_density(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the density of the particles.

        Args:
            clone: If ``True``, return a copy of the density array.

        Returns:
            The particle density array.
        """
        if clone:
            return np.copy(self.density)
        return self.density

    def get_effective_density(self) -> NDArray[np.float64]:
        """Return the effective density of the particles, weighted by the mass.

        Returns:
            Effective particle densities derived from species masses.
        """
        densities = self.get_density()
        if np.size(densities) == 1:
            return np.ones_like(self.get_species_mass()) * densities
        return np.copy(self._data.effective_density[0])

    def get_mean_effective_density(self) -> float:
        """Return the mean effective density of the particles.

        Returns:
            Mean effective density across nonzero-density particles.
        """
        # filter out zero densities for no mass in bin/particle
        effective_density = self.get_effective_density()
        effective_density = effective_density[effective_density != 0]
        if effective_density.size == 0:
            return 0.0
        return float(np.mean(effective_density))

    def get_concentration(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the volume concentration of the particles.

        For ParticleResolved Strategies, this is the number of
        particles per self.volume. Otherwise, it's per 1/m^3.

        Args:
            clone: If ``True``, return a copy of the concentration array.

        Returns:
            Particle concentration in ``1/m^3``.
        """
        concentration = self._data.concentration[0] / self._data.volume[0]
        if clone:
            return np.copy(concentration)
        return concentration

    def get_total_concentration(self, clone: bool = False) -> np.float64:
        """Return the total concentration of the particles.

        Args:
            clone: If ``True``, operate on a copied concentration array.

        Returns:
            Total number concentration in ``1/m^3``.
        """
        return np.sum(self.get_concentration(clone=clone))

    def get_charge(
        self,
        clone: bool = False,
    ) -> NDArray[np.float64] | None:
        """Return the charge per particle.

        Args:
            clone: If ``True``, return a copy of the charge array.

        Returns:
            Particle charge array, or ``None`` when charge tracking is disabled.
        """
        charge = self.charge
        if charge is None:
            return None
        if clone:
            return np.copy(charge)
        return charge

    def get_volume(self, clone: bool = False) -> float:
        """Return the volume used for the particle representation.

        Args:
            clone: If ``True``, return a copy of the volume value.

        Returns:
            Representation volume in ``m^3``.
        """
        if clone:
            return deepcopy(self.volume)
        return self.volume

    def get_species_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the masses per species in the particles.

        Args:
            clone: If ``True``, return a copy of the computed mass array.

        Returns:
            Per-species particle masses in ``kg``.
        """
        masses = self._data.masses[0]
        if self._species_mass_is_1d:
            masses = masses[:, 0]
        if clone:
            return np.copy(masses)
        return masses

    def get_mass(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the mass of the particles as calculated by the strategy.

        Args:
            clone: If ``True``, return a copy of the mass array.

        Returns:
            Total particle masses in ``kg``.
        """
        mass = self._data.total_mass[0]
        if clone:
            return np.copy(mass)
        return mass

    def get_mass_concentration(self, clone: bool = False) -> np.float64:
        """Return the total mass per volume of the simulated particles.

        The mass concentration is calculated from the distribution
        and concentration arrays.

        Args:
            clone: If ``True``, return a copy of the mass concentration value.

        Returns:
            Total particle mass concentration in ``kg/m^3``.
        """
        mass_concentration = np.sum(self.get_mass() * self.get_concentration())
        if clone:
            return deepcopy(mass_concentration)
        return mass_concentration

    def get_radius(self, clone: bool = False) -> NDArray[np.float64]:
        """Return the radius of the particles as calculated by the strategy.

        Args:
            clone: If ``True``, return a copy of the radius array.

        Returns:
            Particle radii in meters.
        """
        radius = self._data.radii[0]
        if clone:
            return np.copy(radius)
        return radius

    def add_mass(self, added_mass: NDArray[np.float64]) -> None:
        """Add mass to the particle distribution and update parameters.

        Args:
            added_mass: Mass increment per distribution bin in ``kg``.
        """
        _warn_deprecated(stacklevel=2)
        distribution, _ = self.strategy.add_mass(
            self.get_distribution(),
            self.get_concentration(),
            self.get_density(),
            added_mass,
        )
        self._update_state(
            distribution=distribution,
            concentration=self._data.concentration[0],
            charge_array=self._charge_array,
        )
        self._enforce_increasing_bins()

    def add_concentration(
        self,
        added_concentration: NDArray[np.float64],
        added_distribution: Optional[NDArray[np.float64]] = None,
        *,
        added_charge: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Add concentration to the particle distribution.

        Args:
            added_concentration: Concentration increment per distribution bin.
            added_distribution: Optional distribution array to merge into the
                current distribution. When omitted, the existing distribution is
                reused.
            added_charge: Optional charge array for newly added particles.

        Raises:
            ValueError: If charge tracking is disabled but the strategy returns
                charge data, or if charge tracking is enabled but no updated
                charge data is returned.
        """
        _warn_deprecated(stacklevel=2)
        # if added_distribution is None, then it will be calculated
        if added_distribution is None:
            message = "Added distribution is value None."
            logger.warning(message)
            added_distribution = self.get_distribution()
        (
            distribution,
            concentration,
            updated_charge,
        ) = self.strategy.add_concentration(
            distribution=self.get_distribution(),
            concentration=self.get_concentration(),
            added_distribution=added_distribution,
            added_concentration=added_concentration,
            charge=self._charge_array,
            added_charge=added_charge,
        )
        if self._charge_is_none:
            if updated_charge is not None:
                raise ValueError(
                    "updated_charge must be None when charge is disabled"
                )
            charge_array = None
        elif updated_charge is None:
            raise ValueError(
                "updated_charge must not be None when charge is set"
            )
        else:
            charge_array = updated_charge
        self._update_state(
            distribution=distribution,
            concentration=concentration,
            charge_array=charge_array,
        )
        self._enforce_increasing_bins()

    def collide_pairs(self, indices: NDArray[np.int64]) -> None:
        """Collide particle pairs for particle-resolved strategies.

        Performs coagulation between particle pairs by delegating to the
        distribution strategy's collide_pairs method. The smaller particle
        (first index in each pair) is merged into the larger particle (second
        index). Mass, concentration, and charge are all updated accordingly.

        Charge conservation is handled automatically: if the particles have
        non-zero charges, they are summed during collisions. This enables
        physically accurate charge conservation in particle-resolved
        coagulation simulations.

        Args:
            indices: Array of particle pair indices with shape ``(K, 2)`` where
                each row is ``[small_index, large_index]``.
        """
        _warn_deprecated(stacklevel=2)
        updated_distribution, updated_concentration, updated_charge = (
            self.strategy.collide_pairs(
                self.distribution,
                self.concentration,
                self.density,
                indices,
                self._charge_array,
            )
        )
        if self._charge_is_none:
            charge_array = None
        elif updated_charge is None:
            charge_array = np.zeros_like(updated_concentration)
        else:
            charge_array = updated_charge
        self._update_state(
            distribution=updated_distribution,
            concentration=updated_concentration,
            charge_array=charge_array,
        )

    def _enforce_increasing_bins(self) -> None:
        """Ensure distribution bins are sorted by increasing radius."""
        distribution, concentration, charge = get_sorted_bins_by_radius(
            radius=self.get_radius(),
            distribution=self.distribution,
            concentration=self.concentration,
            charge=self._charge_array
            if self._charge_array is not None
            else 0.0,
        )
        if self._charge_is_none:
            charge_array = None
        else:
            charge_array = self._coerce_array(
                np.asarray(charge, dtype=np.float64),
                self._data.concentration[0],
            )
        self._update_state(
            distribution=distribution,
            concentration=concentration,
            charge_array=charge_array,
        )
