"""Coagulation strategy module.

Defines an abstract base class and supporting methods for particle
coagulation processes in aerosol simulations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeGuard, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula import gas, particles
from particula.dynamics.coagulation import (
    charged_dimensional_kernel,
    charged_kernel_strategy,
    coagulation_rate,
)
from particula.dynamics.coagulation.particle_resolved_step import (
    particle_resolved_method,
)
from particula.particles import change_particle_representation
from particula.particles.particle_data import ParticleData
from particula.particles.representation import ParticleRepresentation

logger = logging.getLogger("particula")

ParticleLike = Union[ParticleRepresentation, ParticleData]


def _is_particle_data(particle: ParticleLike) -> TypeGuard[ParticleData]:
    """Check whether the particle input is ParticleData."""
    return isinstance(particle, ParticleData)


def _unwrap_particle(particle: ParticleLike) -> ParticleLike:
    """Normalize particle inputs that may be wrapped or proxied.

    Args:
        particle: Particle input to normalize.

    Returns:
        The underlying particle input, unchanged if no wrapper exists.
    """
    return particle


def _get_radius(particle: ParticleLike) -> NDArray[np.float64]:
    """Return particle radii for either legacy or new data containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return particle.radii[0]
    return cast(ParticleRepresentation, particle).get_radius()


def _get_mass(particle: ParticleLike) -> NDArray[np.float64]:
    """Return particle total mass for either legacy or new data containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return particle.total_mass[0]
    return cast(ParticleRepresentation, particle).get_mass()


def _get_concentration(particle: ParticleLike) -> NDArray[np.float64]:
    """Return particle concentration for either legacy or new containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return particle.concentration[0]
    return cast(ParticleRepresentation, particle).concentration


def _get_charge(particle: ParticleLike) -> Optional[NDArray[np.float64]]:
    """Return particle charge for either legacy or new containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return particle.charge[0]
    return cast(ParticleRepresentation, particle).get_charge()


def _get_volume(particle: ParticleLike) -> float:
    """Return particle simulation volume for either legacy or new containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return float(particle.volume[0])
    return cast(ParticleRepresentation, particle).get_volume()


def _get_effective_density(particle: ParticleLike) -> NDArray[np.float64]:
    """Return particle effective density for either legacy or new containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        return particle.effective_density[0]
    return cast(ParticleRepresentation, particle).get_effective_density()


def _get_mean_effective_density(particle: ParticleLike) -> float:
    """Return the mean effective density for either legacy or new containers."""
    particle = _unwrap_particle(particle)
    if isinstance(particle, ParticleData):
        effective_density = particle.effective_density[0]
        effective_density = effective_density[effective_density != 0]
        if effective_density.size == 0:
            return 0.0
        return float(np.mean(effective_density))
    return cast(ParticleRepresentation, particle).get_mean_effective_density()


def _get_particle_resolved_kernel_radius_data(
    data: ParticleData,
    bin_radius: Optional[NDArray[np.float64]] = None,
    total_bins: Optional[int] = None,
    bins_per_radius_decade: int = 10,
) -> NDArray[np.float64]:
    """Determine binned radii for ParticleData kernel calculations.

    Args:
        data: ParticleData input for radius binning.
        bin_radius: Optional array of radius bin edges in meters.
        total_bins: Exact number of bins to generate, if set.
        bins_per_radius_decade: Number of bins per decade of radius.

    Returns:
        Bin edges (radii) in meters.

    Raises:
        ValueError: If finite radii cannot be determined for binning.
    """
    if bin_radius is not None:
        return bin_radius
    particle_radius = data.radii[0]
    positive_mask = particle_radius > 0
    if not np.any(positive_mask):
        raise ValueError(
            "Particle radius must be finite. Check the particles,"
            "they may all be zero and the kernel cannot be calculated."
        )
    min_radius = np.min(particle_radius[positive_mask]) * 0.5
    max_radius = np.max(particle_radius[positive_mask]) * 2
    if not np.isfinite(min_radius) or not np.isfinite(max_radius):
        raise ValueError(
            "Particle radius must be finite. Check the particles,"
            "they may all be zero and the kernel cannot be calculated."
        )
    if min_radius == 0:
        min_radius = np.float64(1e-10)
    if total_bins is not None:
        return np.logspace(
            np.log10(min_radius),
            np.log10(max_radius),
            num=total_bins,
            base=10,
            dtype=np.float64,
        )
    num = np.ceil(
        bins_per_radius_decade * np.log10(max_radius / min_radius),
    )
    return np.logspace(
        np.log10(min_radius),
        np.log10(max_radius),
        num=int(num),
        base=10,
        dtype=np.float64,
    )


def _get_binned_kernel_data_from_particle_data(
    data: ParticleData,
    kernel_radius: NDArray[np.float64],
) -> ParticleData:
    """Bin particle-resolved ParticleData into a kernel-friendly container.

    Args:
        data: ParticleData to bin.
        kernel_radius: Radius bin edges for kernel calculations.

    Returns:
        ParticleData with binned masses, concentration, and charge.
    """
    radii = data.radii[0]
    masses = data.masses[0]
    concentration = data.concentration[0]
    charge = data.charge[0]
    bin_indexes = np.digitize(radii, kernel_radius)
    num_bins = kernel_radius.size
    valid_mask = bin_indexes < num_bins
    if not np.any(valid_mask):
        empty_shape = (1, 0, data.n_species)
        return ParticleData(
            masses=np.zeros(empty_shape, dtype=np.float64),
            concentration=np.zeros((1, 0), dtype=np.float64),
            charge=np.zeros((1, 0), dtype=np.float64),
            density=np.copy(data.density),
            volume=np.array([float(data.volume[0])], dtype=np.float64),
        )
    bin_indexes = bin_indexes[valid_mask]
    masses = masses[valid_mask]
    concentration = concentration[valid_mask]
    charge = charge[valid_mask]
    counts = np.bincount(bin_indexes, minlength=num_bins)
    concentration_sum = np.bincount(
        bin_indexes, weights=concentration, minlength=num_bins
    )
    new_masses = np.zeros((num_bins, data.n_species), dtype=np.float64)
    for species_index in range(data.n_species):
        summed = np.bincount(
            bin_indexes,
            weights=masses[:, species_index],
            minlength=num_bins,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_mass = summed / counts
        mean_mass[counts == 0] = np.nan
        new_masses[:, species_index] = mean_mass
    new_charge = np.full(num_bins, np.nan, dtype=np.float64)
    for bin_index in np.flatnonzero(counts):
        mask = bin_indexes == bin_index
        new_charge[bin_index] = np.median(charge[mask])
    new_charge = np.where(np.isnan(new_charge), 0, new_charge)
    new_concentration = np.where(
        np.isnan(concentration_sum), 0, concentration_sum
    )
    mask_nan_zeros = np.isnan(new_masses) | (new_masses == 0)
    mask_nan_zeros = ~np.any(mask_nan_zeros, axis=1)
    filtered_masses = new_masses[mask_nan_zeros]
    filtered_charge = new_charge[mask_nan_zeros]
    filtered_concentration = new_concentration[mask_nan_zeros]
    return ParticleData(
        masses=filtered_masses[np.newaxis, ...],
        concentration=filtered_concentration[np.newaxis, ...],
        charge=filtered_charge[np.newaxis, ...],
        density=np.copy(data.density),
        volume=np.array([float(data.volume[0])], dtype=np.float64),
    )


def _resolve_radius_indices(
    particle_radius: NDArray[np.float64],
    lookup_values: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Resolve radius values to unique particle indices.

    Args:
        particle_radius: Available particle radii.
        lookup_values: Radii to resolve against particle_radius.

    Returns:
        Index array aligned with lookup_values.

    Raises:
        ValueError: If a lookup value has zero or multiple matches.
    """
    indices = np.empty(lookup_values.size, dtype=np.int64)
    for idx, value in enumerate(lookup_values):
        matches = np.flatnonzero(np.isclose(particle_radius, value))
        if matches.size == 0:
            raise ValueError(f"Direct kernel lookup failed for radius {value}.")
        if matches.size > 1:
            raise ValueError(
                "Direct kernel lookup ambiguous for radius "
                f"{value}; {matches.size} matches found."
            )
        indices[idx] = matches[0]
    return indices


def _build_direct_kernel_index_func(
    particle_radius: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_charge: NDArray[np.float64],
    kernel_builder: Callable[..., NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]:
    """Build an index-based kernel function for particle-resolved sampling."""

    def direct_kernel_index_func(
        small_index: NDArray[np.int64],
        large_index: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        small_arr, large_arr = np.broadcast_arrays(
            np.atleast_1d(small_index), np.atleast_1d(large_index)
        )
        small_flat = small_arr.ravel().astype(np.int64)
        large_flat = large_arr.ravel().astype(np.int64)
        kernel_values = np.empty(small_flat.shape, dtype=np.float64)
        for i, (small_idx, large_idx) in enumerate(
            zip(small_flat, large_flat, strict=True)
        ):
            kernel_matrix = kernel_builder(
                particle_radius=np.array(
                    [
                        particle_radius[small_idx],
                        particle_radius[large_idx],
                    ],
                    dtype=np.float64,
                ),
                particle_mass=np.array(
                    [
                        particle_mass[small_idx],
                        particle_mass[large_idx],
                    ],
                    dtype=np.float64,
                ),
                particle_charge=np.array(
                    [
                        particle_charge[small_idx],
                        particle_charge[large_idx],
                    ],
                    dtype=np.float64,
                ),
                temperature=temperature,
                pressure=pressure,
            )
            kernel_values[i] = kernel_matrix[0, 1]
        return kernel_values.reshape(small_arr.shape)

    return direct_kernel_index_func


class CoagulationStrategyABC(ABC):
    """Abstract base class for defining a coagulation strategy.

    This class defines the methods that must be implemented by any
    coagulation strategy (e.g., for discrete, continuous, or
    particle-resolved distributions).

    Attributes:
        - distribution_type : The type of distribution to be used, one of
          ("discrete", "continuous_pdf", or "particle_resolved").
        - use_direct_kernel : Whether to compute pairwise kernels directly
          for particle-resolved coagulation.

    Methods:
    - dimensionless_kernel : Calculate the dimensionless coagulation kernel.
    - kernel : Obtain the dimensioned coagulation kernel [m^3/s].
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py
        class ExampleCoagulation(CoagulationStrategyABC):
            def dimensionless_kernel(self, diff_kn, coulomb_phi):
                return diff_kn + coulomb_phi
            def kernel(self, particle, temperature, pressure):
                return 1.0
        strategy = ExampleCoagulation("discrete")
        ```

    References:
        Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
        Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    """

    def __init__(
        self,
        distribution_type: str,
        particle_resolved_kernel_radius: Optional[NDArray[np.float64]] = None,
        particle_resolved_kernel_bins_number: Optional[int] = None,
        particle_resolved_kernel_bins_per_decade: int = 10,
        use_direct_kernel: bool = False,
    ):
        """Initialize the coagulation strategy.

        Arguments:
            distribution_type: Type of distribution ("discrete",
                "continuous_pdf", or "particle_resolved").
            particle_resolved_kernel_radius: Kernel radius for
                particle-resolved simulations.
            particle_resolved_kernel_bins_number: Number of bins for
                particle-resolved kernel.
            particle_resolved_kernel_bins_per_decade: Bins per decade for
                particle-resolved kernel (default 10).
            use_direct_kernel: Whether to compute kernel values directly for
                particle-resolved coagulation instead of using interpolation.

        Raises:
            ValueError: If distribution_type is not valid.
        """
        if distribution_type not in [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]:
            raise ValueError(
                "Invalid distribution type. Must be one of 'discrete', "
                + "'continuous_pdf', or 'particle_resolved'."
            )
        self.distribution_type = distribution_type
        self.random_generator = np.random.default_rng()
        # for particle resolved coagulation strategy
        self.particle_resolved_radius = particle_resolved_kernel_radius
        self.particle_resolved_bins_number = (
            particle_resolved_kernel_bins_number
        )
        self.particle_resolved_bins_per_decade = (
            particle_resolved_kernel_bins_per_decade
        )
        self.use_direct_kernel = use_direct_kernel

    @abstractmethod
    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the dimensionless coagulation kernel.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
              [dimensionless].

        Returns:
            - NDArray[np.float64] : Dimensionless kernel for particle
              coagulation.

        Examples:
            ```py
            H = strategy.dimensionless_kernel(kn_array, phi_array)
            # H might be array([...]) representing the dimensionless kernel
            ```
        """

    @abstractmethod
    def kernel(
        self,
        particle: ParticleLike,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation kernel [m^3/s].

        Uses particle attributes (e.g., radius, mass) along with temperature
        and pressure to return a dimensional kernel for coagulation.

        Arguments:
            particle : ParticleRepresentation or ParticleData providing radius
            and concentration.
            - temperature : The temperature in Kelvin [K].
            - pressure : The pressure in Pascals [Pa].

        Returns:
            - float or NDArray[np.float64] : The coagulation kernel [m^3/s].

        Examples:
            ```py
            k_val = strategy.kernel(particle, 298.15, 101325)
            # k_val can be a scalar or array
            ```
        """

    def loss_rate(
        self,
        particle: ParticleLike,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation loss rate [kg/s].

        Arguments:
            particle : The particle data for which the loss rate
            is calculated.
            - kernel : The coagulation kernel [m^3/s].

        Returns:
            - float or NDArray[np.float64] : The loss rate [kg/s].

        Raises:
            - ValueError : If the distribution type is invalid.

        Examples:
            ```py
            loss = strategy.loss_rate(particle, k_val)
            ```
        """
        particle = _unwrap_particle(particle)
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_loss_rate_discrete(
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_loss_rate_continuous(
                radius=_get_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def gain_rate(
        self,
        particle: ParticleLike,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation gain rate [kg/s].

        Arguments:
            particle : The particle data used in the
            calculation.
            - kernel : The coagulation kernel [m^3/s].

        Returns:
            - float or NDArray[np.float64] : The gain rate [kg/s].

        Raises:
            - ValueError : If the distribution type is invalid.

        Examples:
            ```py
            gain = strategy.gain_rate(particle, k_val)
            ```
        """
        particle = _unwrap_particle(particle)
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_gain_rate_discrete(
                radius=_get_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_gain_rate_continuous(
                radius=_get_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def net_rate(
        self,
        particle: ParticleLike,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the net coagulation rate = gain - loss [kg/s].

        Arguments:
            - particle : The particle data.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].

        Returns:
            float or NDArray[np.float64] : The net coagulation rate
            [kg/s]. (positive => net gain, negative => net loss).

        Examples:
            ```py
            net = strategy.net_rate(particle, 298.15, 101325)
            ```
        """
        particle = _unwrap_particle(particle)
        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # Type narrowing: cast to NDArray for method compatibility
        kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
        loss_rate = self.loss_rate(particle=particle, kernel=kernel_arr)
        gain_rate = self.gain_rate(particle=particle, kernel=kernel_arr)
        return gain_rate - loss_rate

    def step(  # noqa: C901
        self,
        particle: ParticleLike,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleLike:
        """Perform a single coagulation step over a specified time interval.

        Updates the particle distribution or representation based on the
        net_rate calculated for the given time_step.

        Arguments:
            - particle : The particle data to update.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].
            - time_step : The timestep over which to integrate [s].

        Returns:
            ParticleRepresentation or ParticleData : Updated after
            this step.

        Raises:
            - ValueError : If the distribution type is invalid or unsupported.

        Examples:
            ```py
            updated_particle = strategy.step(particle, 298.15, 101325, 1.0)
            ```
        """
        particle = _unwrap_particle(particle)
        if self.distribution_type in ["discrete", "continuous_pdf"]:
            net_rate_value = self.net_rate(
                particle=particle,
                temperature=temperature,
                pressure=pressure,
            )
            # Type narrowing: cast to NDArray for add_concentration
            concentration_change = cast(
                NDArray[np.float64], np.atleast_1d(net_rate_value * time_step)
            )
            if isinstance(particle, ParticleData):
                particle.concentration[0] += concentration_change
                return particle
            particle.add_concentration(concentration_change)
            return particle

        if self.distribution_type == "particle_resolved":
            kernel_func: Optional[
                Callable[
                    [NDArray[np.float64], NDArray[np.float64]],
                    NDArray[np.float64],
                ]
            ] = None
            kernel_index_func: Optional[
                Callable[
                    [NDArray[np.int64], NDArray[np.int64]],
                    NDArray[np.float64],
                ]
            ] = None
            if isinstance(particle, ParticleData):
                if (
                    particle.concentration[0].size == 0
                    or np.sum(particle.concentration[0]) == 0
                ):
                    return particle
                kernel_radius = _get_particle_resolved_kernel_radius_data(
                    data=particle,
                    bin_radius=self.particle_resolved_radius,
                    total_bins=self.particle_resolved_bins_number,
                    bins_per_radius_decade=self.particle_resolved_bins_per_decade,
                )
                kernel_data = _get_binned_kernel_data_from_particle_data(
                    data=particle, kernel_radius=kernel_radius
                )
                kernel_radius = _get_radius(kernel_data)
                if not np.any(kernel_radius):
                    raise ValueError(
                        "The kernel radius cannot be zero. "
                        "Check the particle representation."
                    )
                kernel = self.kernel(
                    particle=kernel_data,
                    temperature=temperature,
                    pressure=pressure,
                )
                kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
                sort_indices = np.argsort(kernel_radius)
                kernel_radius = kernel_radius[sort_indices]
                if kernel_arr.ndim == 2:
                    kernel_arr = kernel_arr[sort_indices][:, sort_indices]
                step_func = particle_resolved_method.get_particle_resolved_coagulation_step  # noqa: E501
                if self.use_direct_kernel:
                    kernel_strategy = getattr(self, "kernel_strategy", None)
                    kernel_mapping = {
                        charged_kernel_strategy.HardSphereKernelStrategy: (
                            charged_dimensional_kernel.get_hard_sphere_kernel_via_system_state
                        ),
                        (
                            charged_kernel_strategy.CoulombDyachkov2007KernelStrategy
                        ): (
                            charged_dimensional_kernel.get_coulomb_kernel_dyachkov2007_via_system_state
                        ),
                        (
                            charged_kernel_strategy.CoulombGatti2008KernelStrategy
                        ): (
                            charged_dimensional_kernel.get_coulomb_kernel_gatti2008_via_system_state
                        ),
                        (
                            charged_kernel_strategy.CoulombGopalakrishnan2012KernelStrategy
                        ): (
                            charged_dimensional_kernel.get_coulomb_kernel_gopalakrishnan2012_via_system_state
                        ),
                        (
                            charged_kernel_strategy.CoulumbChahl2019KernelStrategy
                        ): (
                            charged_dimensional_kernel.get_coulomb_kernel_chahl2019_via_system_state
                        ),
                    }
                    kernel_builder = None
                    if kernel_strategy is not None:
                        kernel_builder = kernel_mapping.get(
                            type(kernel_strategy)
                        )
                    if kernel_builder is None:
                        logger.warning(
                            "Direct kernel disabled for unsupported kernel "
                            "strategy."
                        )
                    else:
                        particle_radius = np.asarray(
                            _get_radius(particle), dtype=np.float64
                        )
                        particle_mass = np.asarray(
                            _get_mass(particle), dtype=np.float64
                        )
                        particle_charge = _get_charge(particle)
                        if particle_charge is None:
                            particle_charge = np.zeros_like(
                                particle_radius, dtype=np.float64
                            )
                        charge_array = np.asarray(
                            particle_charge, dtype=np.float64
                        )
                        kernel_index_func = _build_direct_kernel_index_func(
                            particle_radius=particle_radius,
                            particle_mass=particle_mass,
                            particle_charge=charge_array,
                            kernel_builder=kernel_builder,
                            temperature=temperature,
                            pressure=pressure,
                        )
                loss_gain_indices = step_func(
                    particle_radius=_get_radius(particle),
                    kernel=kernel_arr,
                    kernel_radius=kernel_radius,
                    volume=_get_volume(particle),
                    time_step=time_step,
                    random_generator=self.random_generator,
                    kernel_func=kernel_func,
                    kernel_index_func=kernel_index_func,
                )
                if loss_gain_indices.size == 0:
                    return particle
                small_index = loss_gain_indices[:, 0]
                large_index = loss_gain_indices[:, 1]
                small_mass = particle.masses[0][small_index].copy()
                particle.masses[0][small_index] = 0
                np.add.at(particle.masses[0], large_index, small_mass)
                particle.concentration[0][small_index] = 0
                small_charge = particle.charge[0][small_index].copy()
                particle.charge[0][small_index] = 0
                np.add.at(particle.charge[0], large_index, small_charge)
                return particle
            # get the kernel radius
            func = (  # noqa: E501
                change_particle_representation.get_particle_resolved_binned_radius
            )
            kernel_radius = func(
                particle=cast(ParticleRepresentation, particle),
                bin_radius=self.particle_resolved_radius,
                total_bins=self.particle_resolved_bins_number,
                bins_per_radius_decade=self.particle_resolved_bins_per_decade,  # noqa: E501
            )
            # convert particle representation to calculate kernel
            func2 = change_particle_representation.get_speciated_mass_representation_from_particle_resolved  # noqa: E501
            kernel_particle = func2(
                particle=cast(ParticleRepresentation, particle),
                bin_radius=kernel_radius,
            )
            kernel_radius = kernel_particle.get_radius()
            if not np.any(kernel_radius):
                raise ValueError(
                    "The kernel radius cannot be zero. "
                    "Check the particle representation."
                )
            kernel = self.kernel(
                particle=kernel_particle,
                temperature=temperature,
                pressure=pressure,
            )
            # Type narrowing: cast to NDArray for step function
            kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
            # Ensure kernel_radius is monotonically increasing for binning
            # and reorder kernel matrix accordingly
            sort_indices = np.argsort(kernel_radius)
            kernel_radius = kernel_radius[sort_indices]
            if kernel_arr.ndim == 2:
                kernel_arr = kernel_arr[sort_indices][:, sort_indices]
            # calculate step
            step_func = (
                particle_resolved_method.get_particle_resolved_coagulation_step
            )
            if self.use_direct_kernel:
                kernel_strategy = getattr(self, "kernel_strategy", None)
                kernel_mapping = {
                    charged_kernel_strategy.HardSphereKernelStrategy: (
                        charged_dimensional_kernel.get_hard_sphere_kernel_via_system_state
                    ),
                    (
                        charged_kernel_strategy.CoulombDyachkov2007KernelStrategy
                    ): (
                        charged_dimensional_kernel.get_coulomb_kernel_dyachkov2007_via_system_state
                    ),
                    (charged_kernel_strategy.CoulombGatti2008KernelStrategy): (
                        charged_dimensional_kernel.get_coulomb_kernel_gatti2008_via_system_state
                    ),
                    (
                        charged_kernel_strategy.CoulombGopalakrishnan2012KernelStrategy
                    ): (
                        charged_dimensional_kernel.get_coulomb_kernel_gopalakrishnan2012_via_system_state
                    ),
                    (charged_kernel_strategy.CoulumbChahl2019KernelStrategy): (
                        charged_dimensional_kernel.get_coulomb_kernel_chahl2019_via_system_state
                    ),
                }
                kernel_builder = None
                if kernel_strategy is not None:
                    kernel_builder = kernel_mapping.get(type(kernel_strategy))
                if kernel_builder is None:
                    logger.warning(
                        "Direct kernel disabled for unsupported kernel "
                        "strategy."
                    )
                else:
                    particle_radius = np.asarray(
                        _get_radius(particle), dtype=np.float64
                    )
                    particle_mass = np.asarray(
                        _get_mass(particle), dtype=np.float64
                    )
                    particle_charge = _get_charge(particle)
                    if particle_charge is None:
                        particle_charge = np.zeros_like(
                            particle_radius, dtype=np.float64
                        )
                    charge_array = np.asarray(particle_charge, dtype=np.float64)
                    kernel_index_func = _build_direct_kernel_index_func(
                        particle_radius=particle_radius,
                        particle_mass=particle_mass,
                        particle_charge=charge_array,
                        kernel_builder=kernel_builder,
                        temperature=temperature,
                        pressure=pressure,
                    )

            loss_gain_indices = step_func(
                particle_radius=_get_radius(particle),
                kernel=kernel_arr,
                kernel_radius=kernel_radius,
                volume=_get_volume(particle),
                time_step=time_step,
                random_generator=self.random_generator,
                kernel_func=kernel_func,
                kernel_index_func=kernel_index_func,
            )
            cast(ParticleRepresentation, particle).collide_pairs(
                loss_gain_indices
            )
            return particle

        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete', 'continuous_pdf', or"
            " 'particle_resolved'."
        )

    def diffusive_knudsen(
        self,
        particle: ParticleLike,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Calculate the diffusive Knudsen number for each particle.

        The Knudsen number is used to characterize the relative importance of
        diffusion-controlled processes.

        Arguments:
            - particle : The particle data.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].

        Returns:
            - NDArray[np.float64] : Diffusive Knudsen number(s) [dimensionless].

        Examples:
            ```py
            knudsen_nums = strategy.diffusive_knudsen(particle, 298.15, 101325)
            ```
        """
        particle = _unwrap_particle(particle)
        # properties calculation for friction factor
        friction_factor = self.friction_factor(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # coulomb potential ratio
        coulomb_potential_ratio = self.coulomb_potential_ratio(
            particle=particle, temperature=temperature
        )
        return particles.get_diffusive_knudsen_number(
            particle_radius=_get_radius(particle),
            particle_mass=_get_mass(particle),
            friction_factor=friction_factor,
            coulomb_potential_ratio=coulomb_potential_ratio,
            temperature=temperature,
        )  # type: ignore

    def coulomb_potential_ratio(
        self, particle: ParticleLike, temperature: float
    ) -> NDArray[np.float64]:
        """Calculate the Coulomb potential ratio for each particle.

        This ratio characterizes the influence of electrostatic forces on
        coagulation processes.

        Arguments:
            - particle : The particle data.
            - temperature : The gas-phase temperature [K].

        Returns:
            - NDArray[np.float64] : Coulomb potential ratio(s) [dimensionless].

        Examples:
            ```py
            phi = strategy.coulomb_potential_ratio(particle, 298.15)
            ```
        """
        particle = _unwrap_particle(particle)
        charge = _get_charge(particle)
        if charge is None:
            charge = np.zeros_like(_get_radius(particle), dtype=np.float64)
        return particles.get_coulomb_enhancement_ratio(
            particle_radius=_get_radius(particle),
            charge=charge,
            temperature=temperature,
        )  # type: ignore

    def friction_factor(
        self,
        particle: ParticleLike,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute the friction factor for each particle in the aerosol.

        Considers dynamic viscosity, mean free path, and slip correction to
        determine the friction factor [dimensionless].

        Arguments:
            particle : The particle data for which to compute
            friction factor.
            - temperature : Gas temperature [K].
            - pressure : Gas pressure [Pa].

        Returns:
            - NDArray[np.float64] : Friction factor(s) [dimensionless].

        Examples:
            ```py
            fr = strategy.friction_factor(particle, 298.15, 101325)
            ```
        """
        particle = _unwrap_particle(particle)
        # assume standard atmospheric composition
        dynamic_viscosity = gas.get_dynamic_viscosity(temperature=temperature)
        mean_free_path = gas.get_molecule_mean_free_path(
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        knudsen_number = particles.get_knudsen_number(
            mean_free_path=mean_free_path,
            particle_radius=_get_radius(particle),
        )
        slip_correction = particles.get_cunningham_slip_correction(
            knudsen_number=knudsen_number
        )
        return particles.get_friction_factor(
            particle_radius=_get_radius(particle),
            dynamic_viscosity=dynamic_viscosity,
            slip_correction=slip_correction,
        )  # type: ignore
