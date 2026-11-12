"""Coagulation strategy module.

Defines an abstract base class and supporting methods for particle
coagulation processes in aerosol simulations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula import gas, particles
from particula.dynamics.coagulation import (
    coagulation_rate,
)
from particula.particles.particle_data import ParticleData

from ._native_step import binned_kernel_data, kernel_edges, step_native
from ._native_validation import (
    positive_scalar,
    rate_radius,
    validate_particle,
    validate_rate_kernel,
)

logger = logging.getLogger("particula")


def _require_single_box_particle_data(particle: ParticleData) -> ParticleData:
    """Validate that ParticleData uses the supported single-box CPU layout.

    Args:
        particle: ParticleData input to validate.

    Returns:
        The original ParticleData instance when ``n_boxes == 1``.

    Raises:
        ValueError: If ``particle.n_boxes`` is not 1.
    """
    if not isinstance(particle, ParticleData):
        raise TypeError("coagulation material requires ParticleData")
    if particle.distribution_type is not None:
        validate_particle(particle, cast(str, particle.distribution_type))
    if particle.n_boxes != 1:
        raise ValueError(
            "ParticleData must have n_boxes=1 for CPU coagulation "
            "strategies; multi-box execution is unsupported, got "
            f"n_boxes={particle.n_boxes}."
        )
    return particle


def _get_radius(particle: ParticleData) -> NDArray[np.float64]:
    """Return material radii from single-box native storage."""
    return _require_single_box_particle_data(particle).radii[0]


def _get_mass(particle: ParticleData) -> NDArray[np.float64]:
    """Return total per-particle material mass."""
    return _require_single_box_particle_data(particle).total_mass[0]


def _get_concentration(particle: ParticleData) -> NDArray[np.float64]:
    """Interpret explicitly tagged population state as slot number density."""
    particle = _require_single_box_particle_data(particle)
    validate_particle(particle, cast(str, particle.distribution_type))
    return particle.slot_concentration_density[0]


def _get_charge(particle: ParticleData) -> NDArray[np.float64]:
    """Return explicit material charge lanes."""
    return _require_single_box_particle_data(particle).charge[0]


def _get_volume(particle: ParticleData) -> float:
    """Return the explicit single-box volume."""
    return float(_require_single_box_particle_data(particle).volume[0])


def _get_effective_density(particle: ParticleData) -> NDArray[np.float64]:
    """Return per-particle effective material density."""
    return _require_single_box_particle_data(particle).effective_density[0]


def _get_mean_effective_density(particle: ParticleData) -> float:
    """Return arithmetic mean effective density excluding zero-mass slots."""
    effective_density = _get_effective_density(particle)
    effective_density = effective_density[effective_density != 0]
    if effective_density.size == 0:
        return 0.0
    return float(np.mean(effective_density))


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
    return kernel_edges(data, bin_radius, total_bins, bins_per_radius_decade)


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
    return binned_kernel_data(data, kernel_radius)


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
        distribution_type: The type of distribution to be used, one of
            ("discrete", "continuous_pdf", or "particle_resolved").
        use_direct_kernel: Whether to compute pairwise kernels directly
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

        Args:
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

        Args:
            diffusive_knudsen: The diffusive Knudsen number
                [dimensionless].
            coulomb_potential_ratio: The Coulomb potential ratio
                [dimensionless].

        Returns:
            NDArray[np.float64]: Dimensionless kernel for particle
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
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation kernel [m^3/s].

        Uses particle attributes (e.g., radius, mass) along with temperature
        and pressure to return a dimensional kernel for coagulation. CPU
        ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

        Args:
            particle: Explicitly tagged ParticleData providing radius
                and concentration.
            temperature: The temperature in Kelvin [K].
            pressure: The pressure in Pascals [Pa].

        Returns:
            float or NDArray[np.float64]: The coagulation kernel [m^3/s].

        Raises:
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            k_val = strategy.kernel(particle, 298.15, 101325)
            # k_val can be a scalar or array
            ```
        """

    def loss_rate(
        self,
        particle: ParticleData,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation loss rate.

        Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

        CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

        Args:
            particle: The particle data for which the loss rate
                is calculated.
            kernel: The coagulation kernel [m^3/s].

        Returns:
            The loss rate in 1/m^3/s for PMF or 1/m^4/s for radius PDF.

        Raises:
            ValueError: If the distribution type is invalid.
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            loss = strategy.loss_rate(particle, k_val)
            ```
        """
        native_particle = particle
        validate_particle(native_particle, self.distribution_type)
        kernel = validate_rate_kernel(native_particle, kernel)
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_loss_rate_discrete(
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_loss_rate_continuous(
                radius=rate_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def gain_rate(
        self,
        particle: ParticleData,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation gain rate.

        Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

        CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

        Args:
            particle: The particle data used in the
                calculation.
            kernel: The coagulation kernel [m^3/s].

        Returns:
            The gain rate in 1/m^3/s for PMF or 1/m^4/s for radius PDF.

        Raises:
            ValueError: If the distribution type is invalid.
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            gain = strategy.gain_rate(particle, k_val)
            ```
        """
        native_particle = particle
        validate_particle(native_particle, self.distribution_type)
        kernel = validate_rate_kernel(native_particle, kernel)
        if self.distribution_type in ("discrete", "continuous_pdf"):
            if not np.any(native_particle.concentration):
                return np.zeros(native_particle.n_particles, dtype=np.float64)
            radius = rate_radius(native_particle)
            if (
                radius.size < 4
                or np.any(radius <= 0)
                or np.any(np.diff(radius) <= 0)
            ):
                raise ValueError(
                    "coagulation gain requires at least four increasing "
                    "positive radius nodes"
                )
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_gain_rate_discrete(
                radius=rate_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_gain_rate_continuous(
                radius=rate_radius(particle),
                concentration=_get_concentration(particle),
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def net_rate(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the net coagulation rate as gain minus loss.

        Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

        CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

        Args:
            particle: The particle data.
            temperature: The gas-phase temperature [K].
            pressure: The gas-phase pressure [Pa].

        Returns:
            float or NDArray[np.float64] : The net coagulation rate
            in 1/m^3/s for PMF or 1/m^4/s for radius PDF.
            Positive values mean net gain; negative values mean net loss.

        Raises:
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            net = strategy.net_rate(particle, 298.15, 101325)
            ```
        """
        validate_particle(particle, self.distribution_type)
        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # Type narrowing: cast to NDArray for method compatibility
        kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
        loss_rate = self.loss_rate(particle=particle, kernel=kernel_arr)
        gain_rate = self.gain_rate(particle=particle, kernel=kernel_arr)
        return gain_rate - loss_rate

    def step(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleData:
        """Perform a single coagulation step over a specified time interval.

        Update density populations from their net rate or merge selected
        resolved collisions. Explicit distribution metadata and a single box
        are required, including on zero-duration calls.

        Args:
            particle: The particle data to update.
            temperature: The gas-phase temperature [K].
            pressure: The gas-phase pressure [Pa].
            time_step: The timestep over which to integrate [s].

        Returns:
            ParticleData: The identical updated native container.

        Raises:
            TypeError: If particle is not ParticleData.
            ValueError: If distribution metadata is absent or unsupported,
                or if a CPU ParticleData input has ``n_boxes != 1``.

        Examples:
            ```py
            updated_particle = strategy.step(particle, 298.15, 101325, 1.0)
            ```
        """
        if not isinstance(particle, ParticleData):
            raise TypeError("coagulation step requires ParticleData")
        if particle.distribution_type is None:
            raise ValueError(
                "coagulation step requires explicit distribution_type"
            )
        return step_native(self, particle, temperature, pressure, time_step)

    def diffusive_knudsen(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Calculate the diffusive Knudsen number for each particle.

        The Knudsen number is used to characterize the relative importance of
        diffusion-controlled processes. CPU ``ParticleData`` inputs are
        supported only for ``n_boxes == 1``.

        Args:
            particle: The particle data.
            temperature: The gas-phase temperature [K].
            pressure: The gas-phase pressure [Pa].

        Returns:
            NDArray[np.float64]: Diffusive Knudsen number(s) [dimensionless].

        Raises:
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            knudsen_nums = strategy.diffusive_knudsen(particle, 298.15, 101325)
            ```
        """
        particle = _require_single_box_particle_data(particle)
        if particle.distribution_type is not None:
            validate_particle(
                cast(ParticleData, particle), self.distribution_type
            )
            positive_scalar(temperature, "temperature")
            positive_scalar(pressure, "pressure")
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
        self, particle: ParticleData, temperature: float
    ) -> NDArray[np.float64]:
        """Calculate the Coulomb potential ratio for each particle.

        This ratio characterizes the influence of electrostatic forces on
        coagulation processes. CPU ``ParticleData`` inputs are supported only
        for ``n_boxes == 1``.

        Args:
            particle: The particle data.
            temperature: The gas-phase temperature [K].

        Returns:
            NDArray[np.float64]: Coulomb potential ratio(s) [dimensionless].

        Raises:
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            phi = strategy.coulomb_potential_ratio(particle, 298.15)
            ```
        """
        particle = _require_single_box_particle_data(particle)
        if particle.distribution_type is not None:
            validate_particle(
                cast(ParticleData, particle), self.distribution_type
            )
            positive_scalar(temperature, "temperature")
        charge = _get_charge(particle)
        return particles.get_coulomb_enhancement_ratio(
            particle_radius=_get_radius(particle),
            charge=charge,
            temperature=temperature,
        )  # type: ignore

    def friction_factor(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute the friction factor for each particle in the aerosol.

        Considers dynamic viscosity, mean free path, and slip correction to
        determine the friction factor [kg/s]. CPU ``ParticleData``
        inputs are supported only for ``n_boxes == 1``.

        Args:
            particle: The particle data for which to compute
                friction factor.
            temperature: Gas temperature [K].
            pressure: Gas pressure [Pa].

        Returns:
            NDArray[np.float64]: Friction factor(s) [kg/s].

        Raises:
            ValueError: If a CPU ``ParticleData`` input has ``n_boxes != 1``.

        Examples:
            ```py
            fr = strategy.friction_factor(particle, 298.15, 101325)
            ```
        """
        particle = _require_single_box_particle_data(particle)
        if particle.distribution_type is not None:
            validate_particle(
                cast(ParticleData, particle), self.distribution_type
            )
            positive_scalar(temperature, "temperature")
            positive_scalar(pressure, "pressure")
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
